import random
import numpy as np
from equations import *


# 定义无人机个体的类
class DefineAgent:
    def __init__(self, args, agent_num):
        # 定义无人机的基本属性
        self.speed = random.choice(args.speed)  # 无人机单次交互过程中的移动速度
        self.prob_correct = args.prob_correct  # 精确率
        self.prob_false_alarm = args.prob_false_alarm  # 漏警概率
        self.det_range = args.det_range  #

        # 检测范围
        self.com_range = args.com_range  # 通信范围
        self.env_range = args.env_range  # 环境范围
        self.out_penalty = 0  # 定义的离开限定范围的惩罚项
        self.cash_penalty = args.cash_penalty  # 碰撞损失
        self.rem_length = args.rem_length  # 定义的remember长度
        self.out_penalty_index = args.out_penalty_index  # 惩罚项的比例系数

        # 初始化无人机的位置，朝向（位置随机初始化，暂时不考虑朝向）
        self.x_pos = random.randint(0, self.env_range - 1)
        self.y_pos = random.randint(0, self.env_range - 1)
        self.toward = random.randrange(0, 360, 45)

        # 定义无人机对环境信息的map储备
        self.env_shape = (self.env_range, self.env_range)
        # 存在概率地图，初始化为0.5
        self.exist_prob = np.ones(self.env_shape, dtype=float) / 2
        self.exist_prob_pre = np.ones(self.env_shape, dtype=float)
        # 定义概率地图的检测价值
        self.det_value_map = np.ones(self.env_shape, dtype=float)  # 定义单个位置的检测价值
        self.det_total_value_map = np.ones(self.env_shape, dtype=float)  # 定义覆盖面积的检测价值之和

        # 定义中间变量L，作为整个代码的中间流通变量，初始化为0，对应所有的概率均为0.5
        self.com_value_map = np.zeros(self.env_shape, dtype=float)  # 定义单个位置的检测价值
        self.pre_com_value_map = np.zeros(self.env_shape, dtype=float)  # 定义单个位置的检测价值

        # 定义初始不确定度地图为0,记录自己对每个栅格的访问频次.定义为0.01是为了避免做分母时出现0的情况
        self.uncertainty = np.ones(self.env_shape, dtype=float) / 100
        # 定义自身位置地图
        self.position_map = np.zeros(self.env_shape, dtype=int)
        self.update_pos_demon()
        # 用于记录概率最新更新的时间
        self.update_time_map = np.zeros(self.env_shape, dtype=int)
        self.gotten_update_time_map = np.zeros(self.env_shape, dtype=int)

        # 用于voronoi图相关
        self.new_met_agent = set()
        self.old_met_agent = set()
        self.agent_number = agent_num
        self.voronoi_mask = np.ones(self.env_shape, dtype=int)
        self.temp_voronoi_mask = np.zeros(self.env_shape, dtype=int)
        self.distance_map = np.zeros(self.env_shape, dtype=int)

        # 目标最大不确定度
        self.uncertainty_belief = args.uncertainty_belief

        # 是否成功的移动
        self.suc_move = False

        # 贪心收益相关参数
        self.greedy3_L = args.greedy3_L
        self.dis_factor = args.dis_factor
        self.greedy_rewards = [0, 0, 0, 0]
        self.directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]

    # 计算每个方向的无人机收益
    def cal_greedy_reward(self, finished_step, tmp_x, tmp_y, com_map):
        self.greedy_rewards = []
        for direction in self.directions:
            new_x = tmp_x + direction[0]
            new_y = tmp_y + direction[1]
            tmp_com_map = com_map.copy()
            tmp_reward = 0
            for x in range(new_x - self.det_range, new_x + self.det_range + 1):
                for y in range(new_y - self.det_range, new_y + self.det_range + 1):
                    if 0 <= x < self.env_range and 0 <= y < self.env_range:
                        tmp_reward = tmp_reward + log_pro_to_det_value(tmp_com_map[x][y])
                        tmp_com_map[x][y] = 4
            tmp_reward = tmp_reward + self.dis_factor * self.cal_per_greedy_reward(1, new_x, new_y, tmp_com_map)

            self.greedy_rewards.append(tmp_reward)
        return self.greedy_rewards

    # 计算某一方向的无人机收益
    def cal_per_greedy_reward(self, finished_step, tmp_x, tmp_y, tmp_com_map):
        max_reward = 0
        if finished_step == self.greedy3_L:
            return max_reward

        for direction in self.directions:
            tmp_reward = 0
            new_x = tmp_x + direction[0]
            new_y = tmp_y + direction[1]
            for x in range(new_x - self.det_range, new_x + self.det_range + 1):
                for y in range(new_y - self.det_range, new_y + self.det_range + 1):
                    if 0 <= x < self.env_range and 0 <= y < self.env_range:
                        tmp_reward = tmp_reward + log_pro_to_det_value(tmp_com_map[x][y])
                        tmp_com_map[x][y] = 4

            tmp_reward = tmp_reward + self.dis_factor * self.cal_per_greedy_reward(finished_step + 1, new_x, new_y, tmp_com_map)
            max_reward = max(max_reward, tmp_reward)

        return max_reward

    def find_max_direction(self):
        return self.greedy_rewards.index(max(self.greedy_rewards))

    # 更新无人机当前位置状态
    def update_pos_demon(self):
        # 将自己所在的位置标注为1
        self.position_map = np.zeros(self.env_shape, dtype=int)
        if 0 <= self.x_pos < self.env_range and 0 <= self.y_pos < self.env_range:
            self.position_map[self.x_pos, self.y_pos] = 1

    # 更新自己的地图信息
    def update_map(self, targets, time_step, noise):
        # 循环检测区域的每一个栅格
        for i in range(self.x_pos - self.det_range, self.x_pos + self.det_range + 1):
            for j in range(self.y_pos - self.det_range, self.y_pos + self.det_range + 1):
                # 所检测的方格是否是目标区域
                if 0 <= i < self.env_range and 0 <= j < self.env_range:
                    # 判断当前方格是否存在目标
                    whether_exist = 0
                    for target in targets:
                        # 随机判断是否存在目标，而不是准确判断
                        if i == target.x_pos and j == target.y_pos:
                            whether_exist = 1
                            break
                    det_right = random.random()
                    if whether_exist == 1:
                        if det_right < self.prob_correct:
                            whether_exist = 1
                        else:
                            whether_exist = 0
                    else:
                        if det_right < self.prob_false_alarm:
                            whether_exist = 1
                        else:
                            whether_exist = 0

                    # 更新栅格内的存在概率
                    self.exist_prob[i, j] = log_pro_to_exist_pro(self.com_value_map[i, j])
                    self.exist_prob[i, j] = update_exist_prob(self.exist_prob[i, j],
                                                              self.prob_correct - noise[i, j],
                                                              self.prob_false_alarm + noise[i, j], whether_exist)
                    self.com_value_map[i, j] = exist_pro_to_log_pro(self.exist_prob[i, j])
                    # 更新概率信息新鲜度
                    # if self.x_pos == i and self.y_pos == j:
                    #     self.update_time_map[i, j] = time_step
                    # 更新不确定度（访问频次）加一
                    # self.uncertainty[i, j] = self.uncertainty[i, j] + 1

    def agent_move(self, next_move, agents_pos_map):
        # 不带有朝向约束环境
        # 一共为0,1,2,3个方向
        x_plus = 0
        y_plus = 0
        if next_move == 0:
            x_plus += 1
        elif next_move == 1:
            y_plus += 1
        elif next_move == 2:
            x_plus -= 1
        else:
            y_plus -= 1
        if 0 <= x_plus + self.x_pos < self.env_range and 0 <= y_plus + self.y_pos < self.env_range:
            # 看看目标位置是否有无人机，有的话就返回false，然后重新生成动作；没有的话，就移动无人机并将对应位置置为1
            if agents_pos_map[x_plus + self.x_pos][y_plus + self.y_pos] == 1:
                return False
            else:
                self.x_pos = x_plus + self.x_pos
                self.y_pos = y_plus + self.y_pos
                self.suc_move = True
                agents_pos_map[self.x_pos][self.y_pos] = 1
        else:
            self.out_penalty = self.out_penalty_index
            self.suc_move = True
            agents_pos_map[self.x_pos][self.y_pos] = 1

        return True

    # 处理后的访问时间地图
    def get_update_map(self):
        if np.max(self.update_time_map) > self.rem_length:
            self.gotten_update_time_map = self.update_time_map - np.max(self.update_time_map) + self.rem_length
            # self.gotten_update_time_map = np.clip(self.gotten_update_time_map, 0, self.rem_length)
        else:
            self.gotten_update_time_map = self.update_time_map
        return self.gotten_update_time_map

    # 计算收益
    def cal_reward(self, agents):
        # 循环检测区域的每一个栅格
        reward = []
        for i in range(self.x_pos - self.det_range, self.x_pos + self.det_range + 1):
            for j in range(self.y_pos - self.det_range, self.y_pos + self.det_range + 1):
                # 所检测的方格是否是目标区域
                if 0 <= i < self.env_range and 0 <= j < self.env_range:
                    partial_reward = log_pro_to_det_value(self.com_value_map[i, j])
                    reward.append(partial_reward)
        per_reward = sum(reward)
        self.out_penalty = 0
        per_reward += self.out_penalty
        cash_times = -1
        for agent in agents:
            if agent.x_pos == self.x_pos and agent.y_pos == self.y_pos:
                cash_times += 1
        per_reward = per_reward - cash_times * self.cash_penalty  # 暂定为0
        return per_reward

    '''
    # 5.28版本
    def find_max_pos(self):
        # 检测价值地图中最大值对应的位置
        max_value = 0
        max_x = 0
        max_y = 0
        max_pos = []
        max_min_dis = 0x7FFFFFFF
        for i in range(self.env_range):
            for j in range(self.env_range):
                if self.det_value_map[i, j] * self.voronoi_mask[i, j] == max_value:
                    # 最大且最近的一个
                    if max_min_dis > abs(i - self.x_pos) + abs(j - self.y_pos):
                        max_x = i - self.x_pos
                        max_y = j - self.y_pos
                        max_value = self.det_value_map[i, j]
                        max_min_dis = abs(i - self.x_pos) + abs(j - self.y_pos)
                elif self.det_value_map[i, j] * self.voronoi_mask[i, j] > max_value:
                    max_x = i - self.x_pos
                    max_y = j - self.y_pos
                    max_value = self.det_value_map[i, j]

        if (max_x == 0) & (max_y == 0):
            max_x = 0.5
            max_y = 0.5
        if max_value < self.uncertainty_belief:
            self.new_met_agent = set()
            self.old_met_agent = set()
            self.voronoi_mask = np.ones(self.env_shape, dtype=int)
            self.temp_voronoi_mask = np.zeros(self.env_shape, dtype=int)
            max_min_dis = 0x7FFFFFFF
            for i in range(self.env_range):
                for j in range(self.env_range):
                    if self.det_value_map[i, j] * self.voronoi_mask[i, j] == max_value:
                        # 最大且最近的一个
                        if max_min_dis > abs(i - self.x_pos) + abs(j - self.y_pos):
                            max_x = i - self.x_pos
                            max_y = j - self.y_pos
                            max_value = self.det_value_map[i, j]
                            max_min_dis = abs(i - self.x_pos) + abs(j - self.y_pos)
                    elif self.det_value_map[i, j] * self.voronoi_mask[i, j] > max_value:
                        max_x = i - self.x_pos
                        max_y = j - self.y_pos
                        max_value = self.det_value_map[i, j]

            if (max_x == 0) & (max_y == 0):
                max_x = 0.5
                max_y = 0.5

        max_value = max_value / (abs(max_x) + abs(max_y))
        max_value = max(max_value, 0.05)
        if max_x > 0:
            max_x = 1
        elif max_x < 0:
            max_x = -1
        else:
            max_x = 0
        if max_y > 0:
            max_y = 1
        elif max_y < 0:
            max_y = -1
        else:
            max_y = 0

        max_pos.append(max_x)
        max_pos.append(max_y)
        max_pos.append(max_value)

        return max_pos
    '''
    '''
    def find_max_pos(self):
        # 检测价值地图中最大值对应的位置
        max_value = 0x7FFFFFFF
        max_x = 0
        max_y = 0
        max_pos = []
        max_min_dis = 0x7FFFFFFF

        # com_map的版本
        abs_com_value_map = np.absolute(self.com_value_map)
        for i in range(self.env_range):
            for j in range(self.env_range):
                if abs_com_value_map[i, j] == max_value:
                    # 最大且最近的一个
                    if self.voronoi_mask[i, j] == 0:
                        continue
                    if max_min_dis > abs(i - self.x_pos) + abs(j - self.y_pos):
                        max_x = i - self.x_pos
                        max_y = j - self.y_pos
                        max_value = abs_com_value_map[i, j]
                        max_min_dis = abs(i - self.x_pos) + abs(j - self.y_pos)
                elif abs_com_value_map[i, j] < max_value:
                    if self.voronoi_mask[i, j] == 0:
                        continue
                    max_x = i - self.x_pos
                    max_y = j - self.y_pos
                    max_min_dis = abs(i - self.x_pos) + abs(j - self.y_pos)
                    max_value = abs_com_value_map[i, j]

        if (max_x == 0) & (max_y == 0):
            max_x = 0.5
            max_y = 0.5
        # 转换为不确定度
        max_value = log_pro_to_det_value(max_value)

        if max_value < self.uncertainty_belief:
            self.new_met_agent = set()
            self.old_met_agent = set()
            self.voronoi_mask = np.ones(self.env_shape, dtype=int)
            self.temp_voronoi_mask = np.zeros(self.env_shape, dtype=int)
            max_min_dis = 0x7FFFFFFF

            abs_com_value_map = np.absolute(self.com_value_map)
            for i in range(self.env_range):
                for j in range(self.env_range):
                    if abs_com_value_map[i, j] == max_value:
                        # 最大且最近的一个
                        if self.voronoi_mask[i, j] == 0:
                            continue
                        if max_min_dis > abs(i - self.x_pos) + abs(j - self.y_pos):
                            max_x = i - self.x_pos
                            max_y = j - self.y_pos
                            max_value = abs_com_value_map[i, j]
                            max_min_dis = abs(i - self.x_pos) + abs(j - self.y_pos)
                    elif abs_com_value_map[i, j] < max_value:
                        if self.voronoi_mask[i, j] == 0:
                            continue
                        max_x = i - self.x_pos
                        max_y = j - self.y_pos
                        max_min_dis = abs(i - self.x_pos) + abs(j - self.y_pos)
                        max_value = abs_com_value_map[i, j]

            if (max_x == 0) & (max_y == 0):
                max_x = 0.5
                max_y = 0.5

            max_value = log_pro_to_det_value(max_value)

        max_value = max_value / (abs(max_x) + abs(max_y))
        max_value = max(max_value, 0.05)
        if max_x > 0:
            max_x = 1
        elif max_x < 0:
            max_x = -1
        else:
            max_x = 0
        if max_y > 0:
            max_y = 1
        elif max_y < 0:
            max_y = -1
        else:
            max_y = 0

        max_pos.append(max_x)
        max_pos.append(max_y)
        max_pos.append(max_value)

        return max_pos
    '''

    def find_max_pos(self):
        # 检测价值地图中最大值对应的位置
        max_value = 0x7FFFFFFF
        max_x = 0
        max_y = 0
        max_pos = []
        max_min_dis = 0x7FFFFFFF

        # com_map的版本
        abs_com_value_map = np.absolute(self.com_value_map)
        for i in range(self.env_range):
            for j in range(self.env_range):
                if abs_com_value_map[i, j] == max_value:
                    # 最大且最近的一个
                    if self.voronoi_mask[i, j] == 0:
                        continue
                    if max_min_dis > abs(i - self.x_pos) + abs(j - self.y_pos):
                        max_x = i - self.x_pos
                        max_y = j - self.y_pos
                        max_value = abs_com_value_map[i, j]
                        max_min_dis = abs(i - self.x_pos) + abs(j - self.y_pos)
                elif abs_com_value_map[i, j] < max_value:
                    if self.voronoi_mask[i, j] == 0:
                        continue
                    max_x = i - self.x_pos
                    max_y = j - self.y_pos
                    max_min_dis = abs(i - self.x_pos) + abs(j - self.y_pos)
                    max_value = abs_com_value_map[i, j]

        if (max_x == 0) & (max_y == 0):
            max_x = 0.5
            max_y = 0.5
        # 转换为不确定度
        max_value = log_pro_to_det_value(max_value)

        if max_value < self.uncertainty_belief:
            self.new_met_agent = set()
            self.old_met_agent = set()
            self.voronoi_mask = np.ones(self.env_shape, dtype=int)
            self.temp_voronoi_mask = np.zeros(self.env_shape, dtype=int)
            max_min_dis = 0x7FFFFFFF

            abs_com_value_map = np.absolute(self.com_value_map)
            for i in range(self.env_range):
                for j in range(self.env_range):
                    if abs_com_value_map[i, j] == max_value:
                        # 最大且最近的一个
                        if self.voronoi_mask[i, j] == 0:
                            continue
                        if max_min_dis > abs(i - self.x_pos) + abs(j - self.y_pos):
                            max_x = i - self.x_pos
                            max_y = j - self.y_pos
                            max_value = abs_com_value_map[i, j]
                            max_min_dis = abs(i - self.x_pos) + abs(j - self.y_pos)
                    elif abs_com_value_map[i, j] < max_value:
                        if self.voronoi_mask[i, j] == 0:
                            continue
                        max_x = i - self.x_pos
                        max_y = j - self.y_pos
                        max_min_dis = abs(i - self.x_pos) + abs(j - self.y_pos)
                        max_value = abs_com_value_map[i, j]

            if (max_x == 0) & (max_y == 0):
                max_x = 0.5
                max_y = 0.5

            max_value = log_pro_to_det_value(max_value)

        max_value = max_value / (abs(max_x) + abs(max_y))
        max_value = max(max_value, 0.05)
        if max_x > 0:
            max_x = 1
        elif max_x < 0:
            max_x = -1
        else:
            max_x = 0
        if max_y > 0:
            max_y = 1
        elif max_y < 0:
            max_y = -1
        else:
            max_y = 0

        max_pos.append(max_x)
        max_pos.append(max_y)
        max_pos.append(max_value)

        return max_pos


# 定义目标个体的类，位置信息为x,y具体数值
class DefineTarget:
    # 初始化位置
    def __init__(self, args):
        self.env_range = args.env_range
        self.x_pos = random.randint(0, self.env_range - 1)
        self.y_pos = random.randint(0, self.env_range - 1)
