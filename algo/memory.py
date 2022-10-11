from collections import namedtuple
import random
from algo.SumTree import SumTree
import numpy as np

Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards', 'max_pos','next_max_pos'))


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.memory = []
        self.position = 0

        self.e = 0.01
        self.a = 1
        self.a_increment_per_sampling = 0.0002
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.0005

    def __len__(self):
        return len(self.memory)

    def _get_priority(self, error):
        return round((np.abs(error) + self.e) ** self.a, 5)

    def push(self, error, *args):
        p = self._get_priority(error)
        self.tree.add(p, Experience(*args))

    def sample(self, n):
        batch = []
        idxs = []
        # 从TD error的总和进行均分的，但是特殊项是其他项的几十倍，容易使得其被抽取多次，数据大小还是不合适，当数据比较多的时候，容易出现0的情况
        segment = self.tree.total() / n
        priorities = []

        # beta的数值从0经过600次sample逐渐增加到0
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        # a的数值逐渐减小到0.5
        self.a = np.max([0.5, self.a - self.a_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        # beta控制每次sample增加一定的值，也就是说在前600次训练后大概升到1
        is_weight = np.power(self.tree.n_entries * sampling_probabilities + 1e-5, -self.beta)  # （优先级/平均优先级）的负指数幂
        is_weight /= is_weight.max()  # 归一化

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


"""
# 正常的抽取方法
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        # print(len(self.memory),batch_size)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
"""
