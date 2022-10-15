import torch.nn as nn
import torch
import torch.nn.functional as F


# 0501网络结构
class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()

        # 单个比例参数
        self.linear1 = nn.Linear(1, 1, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    # 使用一个元素调用以确定下一个操作，或在优化期间调用batch。返回tensor([[left0exp,right0exp]...]).
    def forward(self, direct_value, pos_value):
        x = pos_value[0]
        y = pos_value[1]
        w = pos_value[2]

        re_val_x = self.linear1(x * w)
        re_val_y = self.linear1(y * w)
        re_val = torch.cat((re_val_x, re_val_y, -re_val_x, -re_val_y), dim=1)

        val = self.sigmoid((re_val + direct_value))
        outs = self.softmax(val)
        return outs


'''

# 除去bn层
class DQNNet(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQNNet, self).__init__()

        # 地图信息的整合
        linear_input_size = w * h * 1
        self.hidden_size1 = 4
        self.linearUp1 = nn.Linear(linear_input_size, self.hidden_size1, bias=False)
        # self.linear1 = nn.Linear(linear_input_size, self.hidden_size1)
        # self.bn1 = nn.BatchNorm1d(self.hidden_size1)

        # 原本收敛的，在全加上bias=false之后不再收敛了，离谱。现在尝试，linearLow3不加bias,其他的还加，看看效果
        # 最大值信息的数值与位置的融合
        self.hiddenSizeLow1 = 4
        self.linearLow1 = nn.Linear(3, self.hiddenSizeLow1)
        # self.bnLow1 = nn.BatchNorm1d(self.hiddenSizeLow1)
        self.hiddenSizeLow2 = 4
        self.linearLow2 = nn.Linear(self.hiddenSizeLow1, self.hiddenSizeLow2)
        # self.bnLow2 = nn.BatchNorm1d(self.hiddenSizeLow2)
        # self.linearLow3 = nn.Linear(self.hiddenSizeLow2, outputs)
        # self.bnLow3 = nn.BatchNorm1d(outputs)

        # 融合处理
        self.linear_cat = nn.Linear(self.hidden_size1 + outputs, outputs)
        # self.bn3 = nn.BatchNorm1d(outputs)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        drop_prob1 = 0.0
        drop_prob2 = 0.0
        self.dropout1 = nn.Dropout(drop_prob1)
        self.dropout2 = nn.Dropout(drop_prob2)

    # 使用一个元素调用以确定下一个操作，或在优化期间调用batch。返回tensor([[left0exp,right0exp]...]).
    def forward(self, x, max_pos):
        # 概率地图和位置地图分别经过两层卷积，然后combine之后再卷积一次
        # 时间经过一层线性层，然后和卷积后的结果combine，然后经过四层线性层，然后softmax
        max_pos = F.relu(self.linearLow1(max_pos))
        max_pos = F.relu(self.linearLow2(max_pos))

        x_split = torch.split(x, 1, dim=1)
        x = x_split[0]
        x = x.reshape(x.size(0), -1)
        x = self.dropout1(F.relu((self.linearUp1(x))))
        x = torch.cat((x, max_pos), dim=1)
        x = self.sigmoid((self.linear_cat(x)))
        outs = self.softmax(x)
        return outs
'''
