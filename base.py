""" 因为item_generate文件中已经有import numpy等，所以使用from item_generate import *，就已经吧numpy等导入进来了"""
from item_genarate import *
import seaborn as sns
import matplotlib.pyplot as plt
import math
from com_type import equations
from com_type.com_type import prop_alg2
from com_type.com_type import prop_alg3


def env_create(args):
    env = RawEnv(args)
    return env


class RawEnv:
    def __init__(self, args):
        # 环境范围、无人机数量、目标数量
        self.env_range = args.env_range
        self.agents_num = args.agents_num
        self.targets_num = args.targets_num
        self.max_step = args.max_step
        self.state_range = args.state_range
        self.time_step = 0

        # 任务终止条件
        self.low_belief = args.low_belief
        self.up_belief = args.up_belief
        self.whether_done = False  # 是否结束
        self.uncertainty_belief = args.uncertainty_belief

        # 决策类型
        self.model_type = args.model_type

        # 是否绘制训练结果的线形图
        self.plot_line = args.plot_line

        # 定义噪声的中间位置以及范围
        self.noise_x_1 = 0
        self.noise_y_1 = 0
        self.noise_x_2 = 0
        self.noise_y_2 = 0
        self.particular_range_1 = args.particular_range_1
        self.particular_range_2 = args.particular_range_2

        # 生成无人机群体于agents
        self.agents = []
        agent_num = 0
        for _ in range(self.agents_num):
            # 包含无人机的基本属性（探测性能、位置朝向、环境初始信息、移动更新、移动后对自身map的更新）,定义部分主要是运行了（探测性能、位置朝向、环境信息）
            per_agent = DefineAgent(args, agent_num)
            agent_num += 1
            self.agents.append(per_agent)

        # 生成目标群体于targets
        self.targets = []
        for _ in range(self.targets_num):
            # 包含目标的位置
            per_target = DefineTarget(args)
            self.targets.append(per_target)

        # 生成一系列地图

        # 生成噪声地图，在后续更新目标存在概率的时候使用
        self.noise_map = np.zeros((self.env_range, self.env_range))
        self.noise_distribution()

        # 错误量地图，用于计算最终结果的好坏
        self.real_pos = np.zeros((self.env_range, self.env_range))
        self.final_error = np.zeros((self.env_range, self.env_range))

        # 记录所有无人机所在位置
        self.agents_pos_map = np.zeros((self.env_range, self.env_range))

        # 保留搜索结果的时长
        self.keep_steps = args.keep_steps

    # 噪声分布
    def noise_distribution(self):
        self.noise_map = np.zeros((self.env_range, self.env_range))
        self.noise_x_1 = random.randint(0, self.env_range - 1)
        self.noise_y_1 = random.randint(0, self.env_range - 1)
        self.noise_x_2 = random.randint(0, self.env_range - 1)
        self.noise_y_2 = random.randint(0, self.env_range - 1)
        for i in range(self.env_range):
            for j in range(self.env_range):
                if math.sqrt(
                        math.pow(i - self.noise_x_1, 2) + math.pow(j - self.noise_y_1, 2)) < self.particular_range_1:
                    self.noise_map[i, j] = 0.2
                elif math.sqrt(
                        math.pow(i - self.noise_x_2, 2) + math.pow(j - self.noise_y_2, 2)) < self.particular_range_2:
                    self.noise_map[i, j] = 0.1

    # 初始化无人机和目标的位置，以及无人机的环境信息
    def reset(self):
        # 用于更新每个栅格的最后搜索时间
        self.time_step = 1
        # 初始化目标所在位置
        for target in self.targets:
            target.x_pos = random.randint(0, self.env_range - 1)
            target.y_pos = random.randint(0, self.env_range - 1)
        # 初始化无人机
        for agent in self.agents:
            # 初始化无人机的位置，朝向
            agent.x_pos = random.randint(0, self.env_range - 1)
            agent.y_pos = random.randint(0, self.env_range - 1)
            agent.toward = random.randrange(0, 360, 45)
            # 如果使用遍历搜索，则起始位置定位（1,0）
            if self.model_type == 'ScanSearch':
                agent.x_pos = 1
                agent.y_pos = 0
            # 存在概率地图，初始化为0.5
            agent.exist_prob = np.ones(agent.env_shape, dtype=float) / 2  # 状态信息中第一项初始化内容：存在概率
            agent.exist_prob_pre = np.ones(agent.env_shape, dtype=float) / 2  # 初始化pre信息
            # 定义初始不确定度地图为0,记录自己对每个栅格的访问频次
            agent.uncertainty = np.ones(agent.env_shape, dtype=float) / 100  # 状态信息中第二项初始化内容：访问次数
            # 初始化com_value_map为0
            agent.com_value_map = np.zeros(agent.env_shape, dtype=float)  # 定义单个位置的检测价值
            # 概率信息新鲜度地图，最新获取该位置访问信息的时间
            agent.update_time_map = np.zeros(agent.env_shape, dtype=int)  # 状态信息中第三项初始化内容：最近访问时间
            # 更新位置地图
            agent.position_map = np.zeros(agent.env_shape, dtype=int)  # 状态信息中第四项初始化内容：更新位置地图
            agent.update_pos_demon()
            # 初始化无人机的位置之后，进行第一次的观测以及环境信息更新
            agent.update_map(self.targets, self.time_step, self.noise_map)
            # 概率信息新鲜度地图
            # agent.update_time_map = np.zeros(agent.env_shape, dtype=int)

            # 初始化voronoi图相关的内容
            agent.new_met_agent = set()
            agent.old_met_agent = set()
            agent.voronoi_mask = np.ones(agent.env_shape, dtype=int)
            agent.temp_voronoi_mask = np.zeros(agent.env_shape, dtype=int)

        # 执行无人机的信息融合
        self.data_fusion()

    # 状态信息获取
    def get_state(self):
        state_list = []
        # 6.5版本，对应cal_voronoi_mask0606使用，计算所有栅格位置到所有无人机的距离
        # self.calDistance()

        for agent in self.agents:
            state = []

            # 搜索价值的两种表征
            '''
            # 5.28版本 不确定概率至探测价值的转换函数
            for i in range(self.env_range):
                for j in range(self.env_range):
                    agent.det_value_map[i, j] = equations.exist_pro_to_det_value(agent.exist_prob[i, j])
            '''
            # 6.5版本
            # for i in range(self.env_range):
            #     for j in range(self.env_range):
            #         agent.det_value_map[i, j] = equations.log_pro_to_det_value(agent.com_value_map[i, j])

            # state.append(self.state_info_cut(agent.det_value_map, agent.x_pos, agent.y_pos))  # 5.28版本
            state.append(self.state_info_cut(agent.com_value_map, agent.x_pos, agent.y_pos))  # 6.5版本
            '''
            # det_all_value_map表征每个位置的搜索价值表征为覆盖位置的价值之和
            for i in range(self.env_range):
                for j in range(self.env_range):
                    agent.det_all_value_map[i, j] = self.pos_reward(i, j, agent.det_value_map)
            state.append(self.state_info_cut(agent.det_all_value_map,agent.x_pos,agent.y_pos))  # 传入整合后的奖励栅格图
            '''
            # stack的用法 https://blog.csdn.net/qq_40605167/article/details/81210926
            states = np.stack(state, axis=0)
            state_list.append(states)

            # 检测价值地图中最大值对应的位置
            # 首先计算无人机遇到的新的其他无人机

            agent.new_met_agent.clear()
            for other_agent in self.agents:
                if other_agent.agent_number == agent.agent_number:
                    continue
                else:
                    if math.hypot(other_agent.x_pos - agent.x_pos, other_agent.y_pos - agent.y_pos) < agent.com_range:
                        agent.new_met_agent.add(other_agent.agent_number)

            # 5.28版本
            # 计算完新遇到的无人机，然后调用相关函数进行处理,利用voronoi_mask函数，判定01，输出哪些是被分配给无人机的.
            # self.cal_voronoi_mask(agent.agent_number)
            agent.voronoi_mask = np.ones(agent.env_shape, dtype=int)
            # 在find_max_pos()，先将区域地图乘以mask，然后再进行计算是否最大值
            max_pos = agent.find_max_pos()

            '''
            # 6.6版本
            # 计算完新遇到的无人机，然后调用相关函数进行处理,利用voronoi_mask函数，判定01，输出哪些是被分配给无人机的.
            self.cal_voronoi_mask0606(agent.agent_number)
            # 在find_max_pos()，先将区域地图乘以mask，然后再进行计算是否最大值
            max_pos = agent.find_max_pos()
            '''

            state_list.append(max_pos)

            # 二进制表征的时间项
            # time_list = dec2bin(self.time_step)
            # state_list.append(time_list)

        return state_list  # 获得下一时刻的环境状态信息，内容为每个无人机的[状态信息（概率地图，位置地图，时间更新地图），二维时间信息]

    # 计算所在位置能记录的覆盖的所有位置的价值之和
    def pos_reward(self, x_pos, y_pos, values):
        pos_rew = 0
        for i in range(x_pos - self.agents[0].det_range, x_pos + self.agents[0].det_range + 1):
            for j in range(y_pos - self.agents[0].det_range, y_pos + self.agents[0].det_range + 1):
                # 所检测的方格是否是目标区域
                if 0 <= i < self.env_range and 0 <= j < self.env_range:
                    pos_rew += values[i, j]
        return pos_rew

    '''
    # 5.28版本
    # 截取探测价值的状态
    def state_info_cut(self, init_map, x_pos, y_pos):
        cut_info = np.ones((2 * self.state_range + 1, 2 * self.state_range + 1))
        for i in range(2 * self.state_range + 1):
            for j in range(2 * self.state_range + 1):
                if 0 <= (x_pos - self.state_range + i) < self.env_range and 0 <= (
                        y_pos - self.state_range + j) < self.env_range:
                    cut_info[i, j] = init_map[x_pos - self.state_range + i, y_pos - self.state_range + j]
                else:
                    cut_info[i, j] = 0
        return cut_info
    '''

    # 6.5版本 截取探测价值的状态
    def state_info_cut(self, init_map, x_pos, y_pos):
        cut_info = np.ones((2 * self.state_range + 1, 2 * self.state_range + 1))
        for i in range(2 * self.state_range + 1):
            for j in range(2 * self.state_range + 1):
                if 0 <= (x_pos - self.state_range + i) < self.env_range and 0 <= (
                        y_pos - self.state_range + j) < self.env_range:
                    cut_info[i, j] = init_map[x_pos - self.state_range + i, y_pos - self.state_range + j]
                    cut_info[i, j] = log_pro_to_det_value(cut_info[i, j])
                else:
                    cut_info[i, j] = 0
        return cut_info

    '''
    # 无人机位置的移动
    def move_agents(self, actions):
        i = 0
        # 每次先将无人机定义为
        self.agents_pos_map = np.zeros((self.env_range, self.env_range))
        for agent in self.agents:
            sucAct = False
            count = 0
            while not sucAct:
                if count > 10:
                    break
                sucAct = agent.agent_move(actions[i], self.agents_pos_map)
                actions[i] = random.randint(0, 3)
                count = count + 1
            i = i + 1
    '''

    # 无人机位置的移动
    def move_agents(self, actions):
        # 每次先将无人机位置地图以及成功的无人机set进行初始化
        self.agents_pos_map = np.zeros((self.env_range, self.env_range))
        Suc_agents = set()
        i = 0
        for agent in self.agents:
            sucAct = agent.agent_move(actions[i], self.agents_pos_map)
            if sucAct:
                Suc_agents.add(i)
            i = i + 1
        count = 0
        while len(Suc_agents) < self.agents_num:
            count = count + 1
            if count > 10:
                break
            for agent in self.agents:
                if not agent.suc_move:
                    action = random.randint(0, 3)
                    sucAct = agent.agent_move(action, self.agents_pos_map)
                    if sucAct:
                        Suc_agents.add(i)

    # 所有agent的奖励的和
    def get_rewards(self, present_step):
        rewards = []
        final_penalty = [0 for i in range(len(self.agents))]
        if present_step == self.max_step - 1:
            for i in range(len(self.agents)):
                final_penalty[i] = self.cal_final_reward(self.agents[i].com_value_map)
        for i in range(len(self.agents)):
            reward = self.agents[i].cal_reward(self.agents) - final_penalty[i]
            rewards.append(reward)
        return rewards

    def render(self):
        plt.ion()
        i = 0

        for agent in self.agents:
            # 绘制每个无人机的目标存在概率地图
            plt.figure(i)
            plt.clf()
            # 调色盘 https://www.cnblogs.com/Forever77/p/11396588.html
            # sns.heatmap https://zhuanlan.zhihu.com/p/165426873

            # 测试生成exist_map的方法
            for i in range(self.env_range):
                for j in range(self.env_range):
                    agent.exist_prob[i, j] = equations.log_pro_to_exist_pro(agent.com_value_map[i, j])
            sns.heatmap(agent.exist_prob.T, vmin=0, vmax=1, linewidths=0.01, annot=False,
                        cmap=sns.color_palette('Blues', self.max_step), fmt=".1f",
                        annot_kws={'color': 'black', "size": 10}).invert_yaxis()
            '''
            sns.heatmap(agent.com_value_map.T, vmin=0, vmax=1, linewidths=0.01, annot=False,
                        cmap=sns.color_palette('Blues', self.max_step), fmt=".1f",
                        annot_kws={'color': 'black', "size": 10}).invert_yaxis()
            '''
            plt.plot(agent.x_pos + 0.5, agent.y_pos + 0.5, marker='p', color='red', markersize=5)
            for target in self.targets:
                plt.plot(target.x_pos + 0.5, target.y_pos + 0.5, marker='*', color='green', markersize=5)
            plt.pause(0.01)
            # 绘制无人机的时间储存地图
            i = i + 1
            break
        """
        # 绘制融合地图
        plt.figure(self.agents_num * 2)
        plt.clf()
        # 调色盘 https://www.cnblogs.com/Forever77/p/11396588.html
        # sns.heatmap https://zhuanlan.zhihu.com/p/165426873
        sns.heatmap(self.fused_exist_prob.T, vmin=0, vmax=1, linewidths=1, annot=True,
                    cmap=sns.color_palette('Blues', self.max_step), fmt=".1f",
                    annot_kws={'color': 'black'}).invert_yaxis()
        for agent in self.agents:
            plt.plot(agent.x_pos + 0.5, agent.y_pos + 0.5, marker='p', color='red', markersize=5)
        for target in self.targets:
            plt.plot(target.x_pos + 0.5, target.y_pos + 0.5, marker='*', color='yellow', markersize=5)
        """
        plt.pause(0.01)

    def render_agent_pos(self):
        plt.figure(-1)
        plt.ion()
        plt.clf()
        for agent in self.agents:
            plt.plot(agent.x_pos + 0.5, agent.y_pos + 0.5, marker='p', color='red', markersize=3)
        plt.xlim(0, self.env_range)
        plt.ylim(0, self.env_range)
        plt.pause(0.00001)

    def render_off(self, episode_rewards, cal_nums):
        plt.ioff()
        if self.plot_line:
            plt.plot(cal_nums, episode_rewards)
            plt.xlabel('episode_num')
            plt.ylabel('rewards')
            plt.show()

    def cal_final_reward(self, com_value_map):
        for i in range(self.env_range):
            for j in range(self.env_range):
                self.final_error[i, j] = equations.log_pro_to_det_value(com_value_map[i, j])
        final_reward = np.sum(self.final_error)
        return final_reward

    # 更新无人机的所有地图信息
    def update_state1(self):
        self.whether_done = False
        for agent in self.agents:
            per_whether_done = True
            # 在每个agent移动之后，更新自己的环境信息
            agent.update_map(self.targets, self.time_step, self.noise_map)  # 存在概率，访问次数，访问时间
            agent.update_pos_demon()  # 存在位置

            # 6.4版本，判断com_map的最小绝对值，对应最大不确定度，是否小于max_unbelief
            abs_com_value_map = np.absolute(agent.com_value_map)
            min_com_L = np.min(abs_com_value_map)
            max_unbelief = equations.log_pro_to_det_value(min_com_L)
            if max_unbelief > self.uncertainty_belief:
                per_whether_done = False

            '''
            # 5.28版本，判断是否搜索结束
            for i in range(self.env_range):
                for j in range(self.env_range):
                    if self.low_belief < agent.exist_prob[i, j] < self.up_belief:
                        per_whether_done = False
            '''
            if per_whether_done:
                self.whether_done = True
                return self.whether_done

    # 更新无人机的所有地图信息
    def update_state2(self):
        # 可通信无人机进行环境信息融合
        self.data_fusion()

        # 根据融合后的结果判断是否结束搜索
        '''
        # 5.28版本，判断是否搜索结束
        for agent in self.agents:
            per_whether_done = True
            for i in range(self.env_range):
                for j in range(self.env_range):
                    if self.low_belief < agent.exist_prob[i, j] < self.up_belief:
                        per_whether_done = False
            if per_whether_done:
                self.whether_done = True
        '''
        # 6.4版本 判断是否结束搜索
        for agent in self.agents:
            per_whether_done = True

            abs_com_value_map = np.absolute(agent.com_value_map)
            min_com_L = np.min(abs_com_value_map)
            max_unbelief = equations.log_pro_to_det_value(min_com_L)
            if max_unbelief > self.uncertainty_belief:
                per_whether_done = False
            if per_whether_done:
                return True

        return self.whether_done

    def get_greedy_rew(self): # 获取窗口式预测算法中不同方向的最大收益


    # 根据通信范围计算无人机之间的邻接特性,然后计算无人机之间的连通性，然后进行存在概率和更新时间两个map的融合
    def data_fusion(self):
        adjacent = np.zeros([self.agents_num, self.agents_num])
        for i in range(self.agents_num):
            for j in range(self.agents_num):
                distance = math.sqrt(
                    math.pow(self.agents[i].x_pos - self.agents[j].x_pos, 2) + math.pow(
                        self.agents[i].y_pos - self.agents[j].y_pos, 2))
                if distance < self.agents[i].com_range:
                    adjacent[i, j] = 1
                    adjacent[j, i] = 1

        # 融合自己范围内的无人机的概率地图
        self.agent_map_fusion(adjacent)

    # 通信范围内无人机地图的融合
    def agent_map_fusion(self, adjacent):
        # 储存当前无人机概率地图——>新版本只需要储存com_value_map就可以
        for agent in self.agents:
            # agent.exist_prob_pre = agent.exist_prob.copy()
            agent.pre_com_value_map = agent.com_value_map.copy()

        # 分别处理无人机的概率地图
        for i in range(self.agents_num):

            # 计算连通的agent
            connected_agent = []
            for j in range(self.agents_num):
                if adjacent[i, j] == 1:
                    connected_agent.append(j)

            # 融合存在概率地图
            # self.prop_alg2_comple(connected_agent, i)
            self.prop_alg2_fast(connected_agent, i)

    # 无人机概率地图的融合公式
    def prop_alg2_comple(self, connected_agent, agent_index):
        count = len(connected_agent)  # 进行加权的无人机数量
        n = 3  # 指数
        exist_prob = np.ones(self.agents[0].env_shape, dtype=float) / 2
        for agent in connected_agent:
            exist_prob = prop_alg2(self.agents[agent].exist_prob_pre, exist_prob, n)
        for k in range(self.env_range):
            for g in range(self.env_range):
                exist_prob[k][g] = math.pow(equations.exist_pro_to_log_pro(exist_prob[k][g]), n) / count
                if exist_prob[k][g] > 0:
                    exist_prob[k][g] = equations.log_pro_to_exist_pro(
                        math.pow(exist_prob[k][g], 1 / n))
                else:
                    exist_prob[k][g] = equations.log_pro_to_exist_pro(
                        -math.pow(-exist_prob[k][g], 1 / n))
        self.agents[agent_index].exist_prob = exist_prob.copy()

    # 无人机概率地图的融合公式
    def prop_alg2_fast(self, connected_agent, agent_index):
        count = len(connected_agent)  # 进行加权的无人机数量
        n = 3  # 指数
        unc_total = np.zeros(self.agents[0].env_shape, dtype=float)
        for agent in connected_agent:
            unc_total = prop_alg3(unc_total, self.agents[agent].pre_com_value_map, n)
        # 进行归一化处理
        unc_total = np.true_divide(unc_total, count)

        # 6.4版本
        sign_np = np.sign(unc_total)
        unc_total = np.absolute(unc_total)
        self.agents[agent_index].com_value_map = np.power(unc_total, 1 / n)
        self.agents[agent_index].com_value_map = np.multiply(self.agents[agent_index].com_value_map, sign_np)

        # 保留探测结果
        for i in range(self.agents[agent_index].x_pos - self.keep_steps - self.agents[agent_index].det_range,
                       self.agents[agent_index].x_pos + self.keep_steps + self.agents[agent_index].det_range + 1):
            for j in range(self.agents[agent_index].y_pos - self.keep_steps - self.agents[agent_index].det_range,
                           self.agents[agent_index].y_pos + self.keep_steps + self.agents[agent_index].det_range + 1):
                # 所检测的方格是否是目标区域
                if 0 <= i < self.env_range and 0 <= j < self.env_range:
                    self.agents[agent_index].com_value_map[i, j] = self.agents[agent_index].pre_com_value_map[i, j]

        '''
        # 6.5版本
        for k in range(self.env_range):
            for g in range(self.env_range):
                if unc_total[k][g] > 0:
                    self.agents[agent_index].com_value_map[k][g] = math.pow(unc_total[k][g], 1 / n)
                else:
                    self.agents[agent_index].com_value_map[k][g] = -math.pow(-unc_total[k][g], 1 / n)
        '''

    # 计算分配的voronoi图
    def cal_voronoi_mask(self, agent_num):
        cur_agent = self.agents[agent_num]
        # 如果注释掉这一行，就是根据最新遇到的无人机进行分配；不注释掉的话，那就是每一步都计算一下通信范围内无人机的voronoi图，但是两种方法都是取交集的策略
        cur_agent.old_met_agent = set()
        cur_agent.new_met_agent = cur_agent.new_met_agent - cur_agent.old_met_agent
        if len(cur_agent.new_met_agent) == 0:
            return
        cur_agent.new_met_agent.add(agent_num)
        cur_agent.old_met_agent = cur_agent.old_met_agent | cur_agent.new_met_agent
        self.agents[agent_num].temp_voronoi_mask = np.zeros(self.agents[agent_num].env_shape, dtype=int)
        # 否则的话
        for i in range(self.env_range):
            for j in range(self.env_range):
                d_min = math.hypot(self.env_range - 1, self.env_range - 1)
                max_agent_num = -1
                # 这个for循环的目标（或者说new_met_agent的定义方法），决定了到底采用那种计算voronoi图的方式
                for agent_order in cur_agent.new_met_agent:
                    d = math.hypot(self.agents[agent_order].x_pos - i, self.agents[agent_order].y_pos - j)
                    if d < d_min:
                        d_min = d
                        max_agent_num = agent_order
                if max_agent_num == agent_num:
                    self.agents[agent_num].temp_voronoi_mask[i][j] = 1
        # 如果打开下面这一步就是经典的voronoi分配方法，否则的话，就是我所设计的方法；如果打开下面这一步，却关掉下下面这一步，那就是不考虑voronoi图的情况
        # self.agents[agent_num].voronoi_mask = np.ones(self.agents[agent_num].env_shape, dtype=int)
        self.agents[agent_num].voronoi_mask = np.multiply(self.agents[agent_num].voronoi_mask,
                                                          self.agents[agent_num].temp_voronoi_mask)

    # 计算分配的voronoi图
    def cal_voronoi_mask0606(self, agent_num):
        cur_agent = self.agents[agent_num]
        # 如果注释掉这一行，就是根据最新遇到的无人机进行分配；不注释掉的话，那就是每一步都计算一下通信范围内无人机的voronoi图，但是两种方法都是取交集的策略
        cur_agent.old_met_agent = set()
        cur_agent.new_met_agent = cur_agent.new_met_agent - cur_agent.old_met_agent
        if len(cur_agent.new_met_agent) == 0:
            return
        cur_agent.new_met_agent.add(agent_num)
        cur_agent.old_met_agent = cur_agent.old_met_agent | cur_agent.new_met_agent
        self.agents[agent_num].temp_voronoi_mask = np.zeros(self.agents[agent_num].env_shape, dtype=int)
        # 将距离信息相并
        stack_list = []
        for agent_order in cur_agent.new_met_agent:
            stack_list.append(self.agents[agent_order].distance_map)

        stack_dis = np.stack(stack_list)
        max_dis = np.min(stack_dis, axis=0)
        self.agents[agent_num].temp_voronoi_mask = (max_dis == self.agents[agent_num].distance_map)
        self.agents[agent_num].temp_voronoi_mask = self.agents[agent_num].temp_voronoi_mask.astype(np.int16)

        # 如果打开下面这一步就是经典的voronoi分配方法，否则的话，就是我所设计的方法；如果打开下面这一步，却关掉下下面这一步，那就是不考虑voronoi图的情况
        self.agents[agent_num].voronoi_mask = np.ones(self.agents[agent_num].env_shape, dtype=int)
        # self.agents[agent_num].voronoi_mask = np.multiply(self.agents[agent_num].voronoi_mask,
        # self.agents[agent_num].temp_voronoi_mask)

    def calDistance(self):
        for i in range(self.env_range):
            for j in range(self.env_range):
                # 计算不同栅格到不同无人机的位置
                for agent in self.agents:
                    agent.distance_map[i, j] = math.hypot(agent.x_pos - i, agent.y_pos - j)


# 时间项的十进制转二进制
def dec2bin(number):
    time_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    num_position = 0
    while number:
        time_list[num_position] = number % 2
        number = number // 2
        num_position += 1
    return time_list
