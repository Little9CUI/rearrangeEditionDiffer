import argparse
import base
import numpy as np
import torch
import numpy
import random
from algo.DQN.dqn_agent import DQN
from equations import exist_pro_to_det_value
from algo.utils import soft_update
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    # 生成随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)

    # 创建强化学习的环境
    env = base.env_create(args)

    # 定义强化学习模型：单智能体强化学习（目前内容）
    action_num = 4  # 动作空间的大小
    model = DQN(env.env_range, action_num, args)  # 生成训练模型
    model.load_model(args.if_train, args.test_num)  # 根据是否需要训练定义加载模型的不同

    # 用于计算loss的平均值
    final_reward = 0
    cal_num = 0
    episode_rewards = []
    cal_nums = []
    per_all_final_reward = 0  # 每个episode结束时，无人机的平均搜索效果

    # 初始化i_episode
    i_episode = 0

    # 标明记录数据的文件
    str_pos = args.save_pos
    with open(str_pos, 'a') as file:
        file.write("\n")

    # 循环次数为max_episodes
    while i_episode < args.max_episodes:

        # 单次episode完成后训练的次数
        if i_episode > 100:
            args.train_times = 400  # 后期train_times的次数要降低，否则训练的次数比交互的次数还要多
        if i_episode > 300:
            args.train_times = 166  # 后期train_times的次数要降低，否则训练的次数比交互的次数还要多

        # 初始化单次episode的 开始时间与结束标志
        start_time = time.time()
        whether_done = False

        env.reset()  # 初始化模型
        next_states = env.get_state()  # 获得下一时刻的环境状态信息，内容为每个无人机的[状态信息（概率地图，位置地图，时间更新地图），二维时间信息]

        for present_step in range(args.max_step):  # 循环无人机移动
            # 是否结束单次搜索
            if whether_done:
                break

            # 仿真绘图
            if args.plot_render & (present_step % 1 == 0):
                env.render()  # 是否渲染
                env.render_agent_pos()

            # 更新基础状态信息值
            env.time_step = env.time_step + 1  # 当前的移动步数
            states = next_states  # 更新状态
            actions = []  # 动作列表

            # 寻找最大窗口收益
            env.get_greedy_reward()

            # 强化学习结果 以及两种贪心策略 以及random策略
            for i in range(len(env.agents)):  # 循环每个无人机
                if args.test_mode == "test" or args.test_mode == "greedy2":
                    action = model.select_action(torch.from_numpy(np.float32(states[2 * i])),
                                                 torch.from_numpy(np.float32(states[2 * i + 1])), args)

                elif args.test_mode == "greedy3":
                    action = env.agents[i].find_max_direction()
                    action = torch.tensor([[action]])

                else:
                    states[2 * i + 1][0] = env.agents[i].x_pos
                    states[2 * i + 1][1] = env.agents[i].y_pos
                    action = model.select_action(torch.from_numpy(env.agents[i].exist_prob),
                                                 np.float32(states[2 * i + 1]), args)
                actions.append(action.squeeze(dim=1))

            actions = torch.cat(actions, dim=0)
            actions = actions.tolist()  # 这两句干什么用的

            # 令无人机执行动作，更新无人机的位置
            env.move_agents(actions)

            # 每个agent得到的奖励
            rewards = env.get_rewards(present_step)

            # 将无人机移动对环境造成的影响进行更新。主要是更新的每个无人机储存的所有环境信息，并且在融合前判断是否搜索结束
            whether_done = env.update_state1()

            # 判断步长是否达到上限
            if present_step == args.max_step - 1:
                whether_done = True

            # 如果单次episode结束，则更新reward
            if whether_done:
                time_reward = 0  # present_step
                rewards = [rewards[i] - time_reward for i in range(len(rewards))]
                # 计算单次episode的表现效果（一共消耗多少步）
                per_all_final_reward = present_step

            # 获取新的状态。状态为每个无人机的存在概率地图和更新时间地图
            next_states = env.get_state()

            # 储存交互信息
            if args.if_train:
                for i in range(len(env.agents)):
                    # 奖励的倍率与偏置
                    re_index = 5
                    reward_sub = 1

                    # 储存强化学习训练所需要的样本信息
                    rw_tensor = torch.tensor([rewards[i] * re_index - reward_sub]).float().to(device)  # size=1
                    ac_tensor = torch.tensor([actions[i]]).to(device)  # size= 1
                    states_tensor = torch.tensor([states[2 * i]]).float().to(device)  # size=1*3*21*21
                    max_tensor = torch.tensor([states[2 * i + 1]]).float().to(device)
                    next_states_tensor = torch.tensor([next_states[2 * i]]).float().to(device)  # size=1*3*21*21
                    next_max_tensor = torch.tensor([next_states[2 * i + 1]]).float().to(device)  # size=1*8

                    # 计算error值
                    error = model.get_td_error(states_tensor, ac_tensor, next_states_tensor, rw_tensor, max_tensor,
                                               next_max_tensor)  # float

                    # 将单条经验归到经验池中
                    model.memory.push(error, states_tensor, ac_tensor, next_states_tensor, rw_tensor, max_tensor,
                                      next_max_tensor)

            # 如果没有结束，则进行信息融合，并判断是否结束单次episode
            if not whether_done:
                whether_done = env.update_state2()
                # 若未结束，则获取新的状态。状态为每个无人机的存在概率地图和更新时间地图
                next_states = env.get_state()

            # 再次判断是否结束单次的搜索
            if whether_done:
                time_reward = present_step
                rewards = [rewards[i] - time_reward for i in range(len(rewards))]

        # 测试情况下，单次交互结束，统计信息，该次训练结束后的结果
        if not args.if_train:
            cal_num += 1
            print('\n', "[Episode %05d] reward %6.4f" % (i_episode, per_all_final_reward))
            final_reward -= per_all_final_reward
            if i_episode % args.print_interval == 0:
                if i_episode == 0:
                    print("[Episode %05d] reward %6.4f" % (i_episode, final_reward))
                else:
                    print("[Episode %05d] reward %6.4f" % (i_episode, final_reward / cal_num))
        else:
            "************* 训练模式代码起始位置 *********************"
            # train模式下
            if i_episode > args.train_in_start:
                for i in range(args.train_times):
                    model.update()
                    args.tau = max(0.02, args.tau - 0.00002)  # 400
                    soft_update(model.policy_net, model.target_net, args.tau)

                if i_episode % args.save_interval == 0:
                    save_path = "./algo/DQN/trained_model/policy_net_" + str(i_episode) + ".pt"
                    torch.save(model.policy_net.state_dict(), save_path, _use_new_zipfile_serialization=False)

            # 输出print_interval个loss的平均值
            cal_num += 1
            # print('\n', "[Episode %05d] reward %6.4f" % (i_episode, per_all_final_reward))
            final_reward -= per_all_final_reward

            if i_episode % args.print_interval == 0:
                if i_episode == 0:
                    print('\n', "[Episode %05d] reward %6.4f" % (i_episode, final_reward))
                    final_reward = final_reward * 3
                else:
                    print('\n', "[Episode %05d] reward %6.4f" % (i_episode, final_reward / args.print_interval))
                    print(model.eps_threshold)
                cal_num = 0
                final_reward = 0

            "************** 训练模式代码终止位置 ***********************************"

        # 更新i_episode、以及模型中记录的已完成轮次model.steps_done
        i_episode += 1
        model.steps_done += 1

        # 计算单轮episode所需要的时间
        # time.sleep(1)
        end_time = time.time()
        print(end_time - start_time)

        # 用于分析强化学习的表现
        # print(i_episode,  )
        # if i_episode % 30 == 0:
        #     with open(Args.save_pos, 'a') as file:
        #         file.write(str(final_reward / cal_num) + " ")
        #     break

    # 结束render
    env.render_off(episode_rewards, cal_nums)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # test是强化学习的算法，greedy是寻找覆盖区域不确定和最大的位置的算法，greedy2是寻找最大不确定度位置的算法,greedy3是找窗口式最优
    parser.add_argument('--test_mode', default="greedy2", type=str, help='test/greedy/random/greedy2/greedy3')
    # 环境相关基本要素
    parser.add_argument('--speed', default=[1, 2, 3, 4], type=list, help='移动速度')
    parser.add_argument('--particular_range_1', default=0, type=int, help='特殊环境(存在噪声)1的范围——0.2概率')
    parser.add_argument('--particular_range_2', default=3, type=int, help='特殊环境（存在噪声）2的范围——0.1概率')
    parser.add_argument('--env_range', default=21, type=int, help='环境的范围')  # 大概133步能够覆盖搜索一遍
    parser.add_argument('--com_range', default=6, type=int, help='可通信的单侧范围')
    parser.add_argument('--det_range', default=1, type=int, help='可检测的单侧范围')
    parser.add_argument('--det_range_list', default=[1, 2], type=list, help='探测范围')
    parser.add_argument('--state_range', default=2, type=int, help='传入状态的单侧范围')
    parser.add_argument('--agents_num', default=3, type=int, help='无人机的数量')
    parser.add_argument('--targets_num', default=50, type=int, help='目标的数量')
    parser.add_argument('--prob_correct', default=0.9, type=float, help='检测到目标的概率')
    parser.add_argument('--prob_false_alarm', default=0.1, type=float, help='虚警概率')
    parser.add_argument('--rem_length', default=3, type=int, help='记忆无人机路径的最大步数，暂时没有使用')
    parser.add_argument('--max_step', default=2000, type=int, help='无人机最大步数')
    parser.add_argument('--out_penalty_index', default=0.0, type=float, help='无人机试图离开环境的惩罚')
    parser.add_argument('--cash_penalty', default=0, type=float, help='碰撞的惩罚')
    parser.add_argument('--low_belief', default=0.1, type=float, help='结束搜索的概率下限')
    parser.add_argument('--up_belief', default=0.9, type=float, help='结束搜索的概率上限,注意上下限之和为1')
    parser.add_argument('--uncertainty_belief', default=0.9, type=float, help='结束搜索的概率上限')
    parser.add_argument('--keep_steps', default=2, type=int, help='保留搜索结果的时长')
    parser.add_argument('--voronoi', default=True, type=bool, help='是否使用voronoi格式')
    parser.add_argument('--plot_render', default=False, type=bool, help='是否交互显示')
    parser.add_argument('--plot_line', default=False, type=bool, help='是否绘制线状图')

    # 强化学习过程相关基本要素
    parser.add_argument('--train_times', default=240, type=int, help='每次交互结束之后训练的次数')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
    parser.add_argument('--algo', default="commnet", type=str, help="commnet/bicnet/maddpg，暂时未使用")
    parser.add_argument('--seed', default=666, type=int, help='随机种子')
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--ReplayMemory_len', default=8192 * 2, type=float, help="经验池的大小")
    parser.add_argument("--save_interval", default=100, type=int, help='模型对的保存间隔')
    parser.add_argument("--print_interval", default=1, type=int, help='episode结果的输出间隔')
    parser.add_argument('--tau', default="1", type=float, help="软更新参数比")
    parser.add_argument('--train_in_start', default=10, type=int, help="开始训练的episodes")
    parser.add_argument('--train_in_num', default=10000, type=int, help="在每步交互内部进行训练的episodes")
    parser.add_argument('--TARGET_UPDATE_in', default=16, type=int, help='在每步交互内部进行训练的episodes中，更新网络的训练间隔')
    # # 起始是20，每次更新target网络加一的话，大概在5000episode左右时，更新间隔增加到100
    parser.add_argument('--TARGET_UPDATE_out', default=1, type=int, help='在每步交互完成后进行训练的episodes，更新网络的训练间隔')

    # 与训练和测试均有关的参数
    parser.add_argument('--max_episodes', default=10000, type=int, help='episode的数量')
    parser.add_argument('--if_train', default=False, type=bool, help="训练还是测试")  # true表示训练，false表示测试
    parser.add_argument('--model_type', default="Random", type=str,
                        help="ScanSearch/Random/None/greedy_policy/greedy_policy2/greedy_policy3，无人机的运行模式,正常训练时设计为None")  # 测试的时候，应该写为
    parser.add_argument('--model_type_begin', default="Random", type=str,
                        help="Random/greedy_policy/greedy_policy2/greedy_policy3，初始探索策略")
    parser.add_argument('--greedy3_L', default=2, type=int, help="滚动贪心搜索的预测长度")
    parser.add_argument('--dis_factor', default=0.8, type=float, help="计算贪心收益时候的折扣因子")

    # 仅仅与测试有关的参数
    parser.add_argument('--test_num', default=1800, type=int, help="测试的模型")  # 如果test_num为0，则表示不进行测试
    parser.add_argument('--save_pos', default="results_greedy3.txt", type=str,
                        help="测试储存位置")  # 如果test_num为0，则表示不进行测试

    """
    关于测试的几点说明：
    1. 测试模式：   将if_train改为False，就可以直接进行测试，不用考虑model_type；通过调节test_num，就可以调整利用的学习结果模型
    2. greedy模式：将model_type和model_type_begin设置为greedy_policy，然后state_range改为环境一半大小
    """

    Args = parser.parse_args()
    Args.uncertainty_belief = exist_pro_to_det_value(Args.up_belief)

    if Args.test_mode == "greedy":  # 找可获得最大收益最大的位置，并且有略微的位置项
        Args.model_type = "greedy_policy"
        Args.model_type_begin = "greedy_policy"
        Args.test_num = 0
    elif Args.test_mode == "greedy2":  # 找可获得不确定度最大的栅格，并且有略微的位置项
        Args.model_type = "greedy_policy2"
        Args.model_type_begin = "greedy_policy2"
        Args.test_num = 0
    elif Args.test_mode == "test":  # 训练得到的结果，通过调节参数test_num进行测试不同的训练结果
        Args.state_range = 2
        Args.model_type = "None"
    else:
        Args.model_type = "Random"
        Args.model_type_begin = "Random"
        Args.test_num = 0

    # 储存位置清空 并 添加标题
    with open(Args.save_pos, 'a') as file:
        file.seek(0)  # 定位
        file.truncate()  # 清空文件
        file.write("*********" + Args.save_pos + "**********")

    main(Args)
