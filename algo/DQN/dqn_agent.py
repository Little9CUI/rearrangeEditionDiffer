import torch
from algo.DQN.network import DQNNet
import torch.optim as optim
from algo.memory import *
import random
import math
import torch.nn.functional as F
import numpy as np
from equations import sigmoid
from equations import exist_pro_to_det_value
from algo.utils import soft_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS_START = 0.6
EPS_END = 0.05  # 改成0.01
EPS_DECAY = 300  # 500个episode之后，对应概率约为0.4


class DQN:

    def __init__(self, env_range, action_num, args):
        # 环境相关
        self.det_range = args.det_range
        self.env_range = args.env_range
        self.state_range = args.state_range
        self.BATCH_SIZE = args.batch_size
        self.model_type = args.model_type
        self.gamma = args.gamma
        self.ReplayMemory_len = args.ReplayMemory_len
        self.action_num = action_num
        self.policy_net = DQNNet(2 * self.state_range + 1, 2 * self.state_range + 1, action_num).to(device)
        self.target_net = DQNNet(2 * self.state_range + 1, 2 * self.state_range + 1, action_num).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，不然的话
        self.policy_net.eval()
        self.target_net.eval()

        self.eps_threshold = 0
        self.TARGET_UPDATE_index = 0
        self.TARGET_UPDATE = args.TARGET_UPDATE_in

        self.model_type_begin = args.model_type_begin

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), 0.001)
        # memory大小一般在10e5到10e6之间
        self.memory = ReplayMemory(self.ReplayMemory_len)

        self.steps_done = 0

    def select_action(self, state, max_pos, args):
        sample = random.random()
        # 是否为训练模式
        if args.if_train:
            self.eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        else:
            self.eps_threshold = 0

        if self.model_type == 'greedy_policy' or self.model_type == 'Random' or self.model_type == 'greedy_policy2' or self.model_type == 'greedy_policy3':
            self.eps_threshold = 1

        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1)将返回每行的最大列值。
                # 最大结果的第二列是找到最大元素的索引，因此我们选择具有较大预期奖励的行动。
                actions = self.policy_net(torch.unsqueeze(state, dim=0), torch.unsqueeze(max_pos, dim=0), )
                action = actions.max(1)[1].view(1, 1)
                return action

        else:
            if args.model_type_begin == 'greedy_policy':
                with torch.no_grad():
                    # t.max(1)将返回每行的最大列值。
                    # 最大结果的第二列是找到最大元素的索引，因此我们选择具有较大预期奖励的行动。
                    action = self.greedy_policy(torch.unsqueeze(state, dim=0), max_pos)
                    return action

            elif args.model_type_begin == 'greedy_policy2':
                with torch.no_grad():
                    # t.max(1)将返回每行的最大列值。
                    # 最大结果的第二列是找到最大元素的索引，因此我们选择具有较大预期奖励的行动。
                    action = self.greedy_policy2(max_pos)
                    return action

            elif args.model_type_begin == 'Random':
                return torch.tensor([[random.randrange(self.action_num)]], device=device,
                                    dtype=torch.long)

    def update(self):
        if self.memory.tree.n_entries < self.BATCH_SIZE:
            return
        # 大概batch normalize
        self.policy_net.train()
        # 获取batch，对应的索引，以及权重
        transitions, idxs, is_weights = self.memory.sample(self.BATCH_SIZE)
        batch = Experience(*zip(*transitions))
        # 分别获取batch不同方面的内容
        # 计算非最终状态，但是实际上我没用到
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_states)), device=device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_states
                                           if s is not None])
        state_batch = torch.cat(batch.states)
        action_batch = torch.cat(batch.actions).unsqueeze(dim=1)
        reward_batch = torch.cat(batch.rewards)
        max_batch = torch.cat(batch.max_pos)
        next_max_batch = torch.cat(batch.next_max_pos)
        # 计算当前时刻的状态—动作预测值
        state_action_values = self.policy_net(state_batch, max_batch).gather(1, action_batch)
        next_action = self.policy_net(non_final_next_states, next_max_batch).max(1)[1].unsqueeze(
            1)  # 预测动作
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states, next_max_batch) \
            .gather(1, next_action).squeeze(1).detach()  # 计算下一时刻的状态—动作预测值
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch  # 计算真实的当前时刻状态—动作价值
        errors = torch.abs(expected_state_action_values.unsqueeze(1) - state_action_values).data.numpy()  # 计算TD误差
        for i in range(self.BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, errors[i][0])
        # 利用权重计算loss
        loss = F.smooth_l1_loss(torch.FloatTensor(is_weights) * expected_state_action_values.unsqueeze(1),
                                torch.FloatTensor(is_weights) * state_action_values)
        # 更新参数
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            # 避免梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # 锁死batch
        self.policy_net.eval()

    # 计算单个样本的TD_error
    def get_td_error(self, states_tensor, ac_tensor, next_states_tensor, rw_tensor, max_tensor, next_max_tensor):
        state_action_values = self.policy_net(states_tensor, max_tensor).gather(1,
                                                                                ac_tensor.unsqueeze(dim=1))
        next_action = self.policy_net(next_states_tensor, max_tensor).max(1)[1].unsqueeze(1)
        next_state_values = self.target_net(next_states_tensor, next_max_tensor).gather(1,
                                                                                        next_action).squeeze(
            1).detach()
        expected_state_action_values = (next_state_values * self.gamma) + rw_tensor
        loss = F.smooth_l1_loss(expected_state_action_values.unsqueeze(1), state_action_values).detach().tolist()
        return loss

    def load_model(self, if_train, test_num):
        if if_train:
            return
        else:
            if test_num == 0:
                return
            read_path = "./algo/DQN/trained_model/policy_net_" + str(test_num) + ".pt"
            self.policy_net.load_state_dict(torch.load(read_path))
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # 贪婪搜索对应的自定义函数
    def greedy_policy(self, states, max_pos):
        x_pos = int(max_pos[0])
        y_pos = int(max_pos[1])
        states = states.squeeze().numpy()
        shape = states.shape
        values = np.zeros(shape, dtype=float)
        for k in range(shape[0]):
            for g in range(shape[1]):
                dis = math.sqrt(math.pow(x_pos - k, 2) + math.pow(y_pos - g, 2))
                # dis = math.log(1 * dis+1)
                values[k, g] = -1e-10 * dis + self.pos_reward(k, g, states)
        values[x_pos, y_pos] = -1
        max_pos = np.where(values == np.max(values))
        x_max = max_pos[0][0]
        y_max = max_pos[1][0]
        action_value_list = [math.sqrt(math.pow(x_pos + 1 - x_max, 2) + math.pow(y_pos - y_max, 2)),
                             math.sqrt(math.pow(x_pos - x_max, 2) + math.pow(y_pos + 1 - y_max, 2)),
                             math.sqrt(math.pow(x_pos - 1 - x_max, 2) + math.pow(y_pos - y_max, 2)),
                             math.sqrt(math.pow(x_pos - x_max, 2) + math.pow(y_pos - 1 - y_max, 2))]
        action = action_value_list.index(min(action_value_list))
        action = torch.tensor([[action]])
        return action

    # 贪婪搜索2对应的自定义函数
    def greedy_policy2(self, max_pos):
        x_max = int(max_pos[0])
        y_max = int(max_pos[1])
        if x_max > 0:
            action = 0
        elif x_max < 0:
            action = 2
        else:
            if y_max > 0:
                action = 1
            else:
                action = 3
        action = torch.tensor([[action]])
        return action

    # 用于greedy_policy中进行调用
    def pos_reward(self, x_pos, y_pos, states):
        pos_rew = 0
        for i in range(x_pos - self.det_range, x_pos + self.det_range + 1):
            for j in range(y_pos - self.det_range, y_pos + self.det_range + 1):
                # 所检测的方格是否是目标区域
                if 0 <= i < self.env_range and 0 <= j < self.env_range:
                    pos_rew += exist_pro_to_det_value(states[i, j])
        return pos_rew
