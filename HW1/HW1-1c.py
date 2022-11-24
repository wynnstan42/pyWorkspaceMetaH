import numpy as np
import math

# Step 0: 設定各個參數值
MAXIT = 400  # maximal iteration number -> 主要最大迴圈數
K = 1.0  # Boltzmann rate -> 影響在新的解得分未較好時，仍然可轉移狀態的機率
DWELL = 20  # 計算平衡狀態時需要的迴圈數目 -> 每階段的熱平衡迴圈數
T_high = 1000.0  # 初始溫度 -> 較高的溫度會有較高的轉移機率
T_scale = 0.9  # 每階段降溫比率: t0 >> t0*r >> t0*r*r >> ... -> 介於 0~1 (!=0)，越低則降溫愈快
T_low = 1.0  # 最終冷卻溫度溫度 -> 達此溫度則迴圈終止

# Step 1: 定義問題
NUM_CITY = 9  # 城市數目 -> TSP 問題中的城市節點數量
# 城市之間的 cost -> 從 TSP 問題的網路圖中得到的各節點之間的移動成本的矩陣形式
M = 9999
cost = [
    [M, 2, M, M, M, 7, 3, M, M],
    [2, M, 4, M, M, M, 6, M, M],
    [M, 4, M, 2, M, M, M, 2, M],
    [M, M, 2, M, 1, M, M, 8, M],
    [M, M, M, 1, M, 6, M, M, 2],
    [7, M, M, M, 6, M, M, M, 5],
    [3, 6, M, M, M, M, M, 3, 1],
    [M, M, 2, 8, M, M, 3, M, 4],
    [M, M, M, M, 2, 5, 1, 4, M]
]


# Step 3: 設定目標函式
def SAfunc(x):  # 依據題目，定義出 cost 的計算 function -> fitness function
    tmp_cost = 0
    for i in range(NUM_CITY - 1):   # 遞迴 list 中各節點，加總總成本
        tmp_cost += cost[x[i]][x[i + 1]]
    tmp_cost += cost[x[NUM_CITY - 1]][x[0]]     # 因為要回到起點，故還要加上最後一站到第一站之間的成本
    return tmp_cost


# Step 4: 設定初始解
np.random.seed(0)  # 設定 random seed -> 常為測試用，如要得到較隨機的結果則應將該行註解
x = np.random.permutation(range(NUM_CITY))  # 令 x 設定為 0, ..., (NUM_CITY-1) 的一個隨機排列 -> 初始解應為各城市的隨機排序陣列
# 預先設定 xbest 及 ybest -> 在開始運算前，先定義出目前最佳解 xbest 以及目前最佳得分 ybest
xbest = x   # 預設目前最佳解為初始解
ybest = y = SAfunc(x)   # 利用已定義好的 fitness function 來得出初始解的得分，當作目前最佳得分

# Step 5: 執行 Simulated Annealing
num_it = 0  # 設定起始 iteration = 0
t = T_high  # 設定起始溫度 t = 一開始設定的最高溫度

# Step 5.1: 在限制的最大迴圈數以及一開始定義的最低溫度間進行遞迴計算
while num_it < MAXIT and t > T_low:     # 若達最大遞迴數或溫度已達最低溫度則終止迴圈計算

    for i in range(DWELL):  # Step 5.2 在一開始設定的熱平衡次數中遞迴，不斷嘗試

        # Step 5.2.1: 找鄰居 xnew -> 透過隨機替換兩城市排序位置取得鄰居解
        xnew = x.copy()     # 先令 xnew[] = x[]
        j = np.random.choice(NUM_CITY, 2)   # 任選兩個整數 j[0], j[1] (不可等於0)
        xnew[j[0]], xnew[j[1]] = xnew[j[1]], xnew[j[0]]     # 互換 xnew[j[0]] 和 xnew[j[1]] -> 隨機交換
        ynew = SAfunc(xnew)     # 計算鄰居解的適應度值

        # Step 5.2.2: 判斷鄰居解的適應度值(得分)是否較高，來決定是否要更換目前解
        if ynew < y:  # keep xnew if energy is reduced -> 若鄰居解的成本較低，則替換成鄰居解
            x = xnew.copy()     # 令 x[] = xnew[]
            y = ynew    # 將目前得分改成新的解的得分

            # Step 5.2.3: 判斷新的解是否有較目前最佳解好，來決定是否要更換目前最佳解
            if y < ybest:   # 若新的解的成本較目前最佳解的成本低，則將目前最佳解替換成新的解
                xbest = x.copy()    # 令 xbest[] = x[]
                ybest = y   # 將目前最佳得分改成新的解的得分

        else:
            # Step 5.2.4: keep xnew with probability p -> 增加隨機性，故利用波茲曼隨機分布來模擬隨機行為，決定就算新的解沒有比較好，是否仍會接受新的解
            if np.random.uniform(0.0, 1.0) < math.exp(- (ynew - y) / (K * t)):  # 若隨機生成介於 0~1 的實數比原先能量低，則接受新的解
                x = xnew.copy()     # 令 x[] = xnew[]
                y = ynew    # 將目前得分改成新的解的得分

        print('Estimated minumum at: ', xbest)  # 每次遞迴都顯示目前最佳解
        print('\tfit = %d\n' % (ybest))     # 每次遞迴都顯示目前最佳得分

    t *= T_scale    # 每次的熱平衡後，都應該繼續降溫 -> 降溫速度取決於一開始設定的每階段降溫比率
    num_it += 1     # 每次熱平衡後，都應增加目前遞迴次數
