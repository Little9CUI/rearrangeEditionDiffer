import numpy as np
from com_type import equations
import math


def min_com(unc_1, unc_2):
    unc = np.ones([len(unc_1[0]), len(unc_1[1])])
    for i in range(len(unc_1[0])):
        for j in range(len(unc_1[1])):
            if equations.exist_pro_to_det_value(unc_1[i][j]) < equations.exist_pro_to_det_value(unc_2[i][j]):
                unc[i][j] = unc_1[i][j]
            else:
                unc[i][j] = unc_2[i][j]
    return unc


def prob_equal(unc_1, unc_2):
    unc = (unc_1 + unc_2) / 2
    return unc


# log形式的概率进行取平均
def prop_alg1(unc_1, unc_2):
    unc = np.ones([len(unc_1[0]), len(unc_1[1])])
    for i in range(len(unc_1[0])):
        for j in range(len(unc_1[1])):
            unc[i][j] = equations.exist_pro_to_log_pro(unc_1[i][j]) + equations.exist_pro_to_log_pro(unc_2[i][j])
            unc[i][j] = equations.log_pro_to_exist_pro(unc[i][j])
    return unc


# log形式的概率,先进性n次幂，再进行取平均
def prop_alg2(unc_1, unc_2, n):
    unc = np.ones([len(unc_1[0]), len(unc_1[1])])
    for i in range(len(unc_1[0])):
        for j in range(len(unc_1[1])):
            unc[i][j] = math.pow(equations.exist_pro_to_log_pro(unc_1[i][j]), n) + math.pow(
                equations.exist_pro_to_log_pro(unc_2[i][j]), n)
            if unc[i][j] > 0:
                unc[i][j] = equations.log_pro_to_exist_pro(math.pow(unc[i][j], 1 / n))  # 负数不能开方
            else:
                unc[i][j] = equations.log_pro_to_exist_pro(-math.pow(-unc[i][j], 1 / n))  # 负数不能开方
    return unc


# log形式的概率,先进性n次幂，再进行取平均,但是不进行n次幂的开方
def prop_alg3(unc_total, unc, n):
    # numpy运算形式
    unc_total = unc_total + np.power(unc, n)

    # for循环格式
    # for i in range(len(unc_total[0])):
    #     for j in range(len(unc_total[1])):
    #         unc_total[i][j] = unc_total[i][j] + math.pow(unc[i][j], n)

    return unc_total
