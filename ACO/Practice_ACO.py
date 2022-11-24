# Problem : Max y = x_1^2 + x_2^2 + x_3^3 + x_4^4 where x in [1,30]
# Solution: x_1 = x_2 = x_3 = x_4 = 30
import numpy
import numpy as np
import matplotlib.pyplot as plt

# 參數
rou = 0.8  # 費洛蒙揮發係數
Q = 1  # 費洛蒙總量

NGEN = 100  # 迴圈數
popsize = 100  # 螞蟻數
dimension = 30  # 維度

low = []  # 各變數x下界
up = []  # 各變數x上界

for i in range(dimension):
    low.append(-500)
    up.append(500)


# np.random.seed(0)          # 若要每次跑得都不一樣的結果，就把這行註解掉

class ACO:
    def __init__(self, parameters):
        """
        Ant Colony Optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        # 初始化
        self.NGEN = parameters[0]  # 迴圈數
        self.pop_size = parameters[1]  # 螞蟻數
        self.var_num = len(parameters[2])  # 變數個數
        self.bound = []  # 變數的約束範圍
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])

        self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有螞蟻的位置
        self.g_best = np.zeros((1, self.var_num))  # 全域螞蟻最優的位置

        # 初始化第0代初始全域最優解
        temp = -1
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = np.random.uniform(self.bound[0][j], self.bound[1][j])
            fit = self.fitness(self.pop_x[i])
            if fit > temp:
                self.g_best = self.pop_x[i]
                temp = fit

    def fitness(self, ind_var):
        """
        個體適應值計算
        """
        for i in ind_var:
            y = 0
            y += (-i * numpy.sin(numpy.sqrt(abs(i))))
        return -y

    def actual_fitness(self, ind_var):
        """
        實際個體適應值計算
        """
        for i in ind_var:
            y = 0
            y = y + (-i * numpy.sin(numpy.sqrt(abs(i))))
        return y

    def update_operator(self, gen, t, t_max):
        """
        更新運算元：根據機率更新下一時刻的位置
        """
        lamda = 1 / gen
        pi = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            for j in range(self.var_num):
                pi[i] = (t_max - t[i]) / t_max
                # 更新位置
                if pi[i] < np.random.uniform(0, 1):
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * lamda
                else:
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * (
                            self.bound[1][j] - self.bound[0][j]) / 2
                # 越界保護
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            # 更新t值
            t[i] = (1 - rou) * t[i] + Q * self.fitness(self.pop_x[i])
            # 更新全域最優值
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]
        t_max = np.max(t)
        return t_max, t

    def main(self):
        popobj = []
        best = np.zeros((1, self.var_num))[0]
        for gen in range(1, self.NGEN + 1):
            if gen == 1:
                tmax, t = self.update_operator(gen, np.array(list(map(self.fitness, self.pop_x))),
                                               np.max(np.array(list(map(self.fitness, self.pop_x)))))
            else:
                tmax, t = self.update_operator(gen, t, tmax)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen)))
            print(self.g_best)
            print(self.fitness(self.g_best))
            if self.fitness(self.g_best) > self.fitness(best):
                best = self.g_best.copy()
            print('最好的位置：{}'.format(best))
            print('最小的函數值：{}'.format(self.actual_fitness(best)))
        print("---- End of (successful) Searching ----")

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(1, self.NGEN + 1)]
        plt.plot(t, popobj, color='b', linewidth=2)
        plt.show()


parameters = [NGEN, popsize, low, up]
aco = ACO(parameters)  # 以參數 parameters 來建立一個 class ACO 的物件，叫aco
aco.main()  # 呼叫 aco 的 main 方法
