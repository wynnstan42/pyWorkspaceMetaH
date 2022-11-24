# 用來求下列函數的最小值
import sys

import numpy as np
import math

MAXIT = 400  # maximal iteration number
K = 0.8  # Boltzmann rate
DWELL = 20  # 計算平衡狀態時需要的迴圈數目
T_high = 1000.0  # 初始溫度
T_scale = 0.9  # 演算法每階段降溫比率: t0 --> t0*r --> t0*r*r --> ...
T_low = 1.0  # 最終冷卻溫度溫度
cost_matrix = [[0, 12, 1, 8],
               [12, 0, 2, 3],
               [1, 2, 0, 10],
               [8, 3, 10, 0]]

# 設定目標函式
def SAfunc(arr):
    cost = 0
    for i in range(len(arr)-1):
        cost += cost_matrix[arr[i]][arr[i + 1]]
    cost += cost_matrix[arr[len(arr)-1]][arr[0]]
    return cost  # minimize the objective cost function

#  ==== 主程式 ====
np.random.seed(0)  # 若要每次跑得都不一樣的結果，就把這行註解掉

# 找初始解 x
init_arr = [0, 1, 2, 3]
np.random.shuffle(init_arr)
xbest = x = init_arr
ybest = y = SAfunc(x)  # 算 cost function

num_it = 0
t = T_high

while num_it < MAXIT and t > T_low:

    for i in range(DWELL):
        # 找鄰居 xnew
        random_index1 = np.random.randint(len(init_arr) - 1)
        random_index2 = np.random.randint(len(init_arr) - 1)
        while random_index1 == random_index2:
            random_index2 = np.random.randint(len(init_arr) - 1)

        xnew = x.copy()
        xnew[random_index1], xnew[random_index2] = xnew[random_index2], xnew[random_index1]
        ynew = SAfunc(xnew)

        if ynew < y or np.random.uniform(0.0, 1.0) < math.exp(- (ynew - y) / (K * t)):
            x = xnew
            y = ynew

        if ynew < ybest:  # 若新的成本比較小，取代最佳解
            xbest = xnew
            ybest = ynew

    print('\tf(%s) = %f\n' % (xbest, ybest))

    t *= T_scale
    num_it += 1
