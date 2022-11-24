# 用來求下列函數的最小值
# f(x) = 3 x^4 - 8 x^3 - 6 x^2 + 24 x
# 此函數之 global opt soln = (-1, -19), local opt soln = (2, 8)

import numpy as np
import math

MAXIT = 400  # maximal iteration number
K = 0.8  # Boltzmann rate
DWELL = 20  # 計算平衡狀態時需要的迴圈數目
T_high = 1000.0  # 初始溫度
T_scale = 0.9  # 演算法每階段降溫比率: t0 --> t0*r --> t0*r*r --> ...
T_low = 1.0  # 最終冷卻溫度溫度


# 設定目標函式
def SAfunc(x):
    return (((3 * x - 8) * x - 6) * x + 24) * x  # minimize the objective cost function
    # global opt soln = (-1, -19), local opt soln = (2, 8)


#  ==== 主程式 ====
np.random.seed(0)  # 若要每次跑得都不一樣的結果，就把這行註解掉

# 找初始解 x
xbest = x = np.random.uniform(-3.0, 3.0)  # 隨機給 -3 ~ +3 的小數給 x
ybest = y = SAfunc(x)  # 算 cost function

num_it = 0
t = T_high

while num_it < MAXIT and t > T_low:

    for i in range(DWELL):
        # 找鄰居 xnew
        xnew = x + np.random.uniform(-0.1, 0.1)  # a random real number between -0.1 and 0.1
        ynew = SAfunc(xnew)

        if ynew < y or np.random.uniform(0.0, 1.0) < math.exp(- (ynew - y) / (K * t)):
            x = xnew
            y = ynew

        if ynew < ybest:  # 若新的成本比較小，取代最佳解
            xbest = xnew
            ybest = ynew

    print('\tf(%f) = %f\n' % (xbest, ybest))

    t *= T_scale
    num_it += 1

