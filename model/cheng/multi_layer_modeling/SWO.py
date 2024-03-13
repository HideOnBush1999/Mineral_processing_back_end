import numpy as np
from scipy.special import gamma

def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = np.size(ub)
    Positions = np.zeros((SearchAgents_no, dim))
    if Boundary_no == 1:
        Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    else:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.uniform(0, 1, SearchAgents_no) * (ub_i - lb_i) + lb_i
    return Positions

def Levy(d):
    beta = 3/2
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, d)
    v = np.random.normal(0, 1, d)
    step = u / np.abs(v) ** (1 / beta)
    L = 0.05 * step
    return L

def SWO(SearchAgents_no, Tmax, ub, lb, dim, fobj):
    Best_SW = np.zeros(dim)
    Best_score = float('inf')
    Convergence_curve = np.zeros(Tmax)
    TR = 0.3
    Cr = 0.2
    N_min = 20
    Positions = initialization(SearchAgents_no, dim, ub, lb)
    t = 0
    SW_Fit = np.zeros(SearchAgents_no)

    for i in range(SearchAgents_no):
        SW_Fit[i] = fobj(Positions[i, :])
        if SW_Fit[i] < Best_score:
            Best_score = SW_Fit[i]
            Best_SW = Positions[i, :]

    # 设置收敛曲线的初始值为初始种群的最佳目标函数值
    Convergence_curve[0] = Best_score

    while t < Tmax:
        a = 2 - 2 * (t / Tmax)
        a2 = -1 + -1 * (t / Tmax)
        k = (1 - t / Tmax)
        JK = np.random.permutation(SearchAgents_no)
        if np.random.rand() < TR:
            for i in range(SearchAgents_no):
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                p = np.random.rand()
                C = a * (2 * r1 - 1)
                l = (a2 - 1) * np.random.rand() + 1
                L = Levy(1)
                vc = np.random.uniform(-k, k, dim)
                rn1 = np.random.randn()
                O_P = Positions[i, :].copy()

                for j in range(dim):
                    if i < k * SearchAgents_no:
                        if p < (1 - t / Tmax):
                            if r1 < r2:
                                m1 = np.abs(rn1) * r1
                                Positions[i, j] = Positions[i, j] + m1 * (Positions[JK[0], j] - Positions[JK[1], j])
                            else:
                                B = 1 / (1 + np.exp(l))
                                m2 = B * np.cos(l * 2 * np.pi)
                                Positions[i, j] = Positions[JK[i], j] + m2 * (lb[j] + np.random.rand() * (ub[j] - lb[j]))
                        else:
                            if r1 < r2:
                                Positions[i, j] = Positions[i, j] + C * np.abs(2 * np.random.rand() * Positions[JK[2], j] - Positions[i, j])
                            else:
                                Positions[i, j] = Positions[i, j] * vc[j]
                    else:
                        if r1 < r2:
                            Positions[i, j] = Best_SW[j] + np.cos(2 * l * np.pi) * (Best_SW[j] - Positions[i, j])
                        else:
                            Positions[i, j] = Positions[JK[0], j] + r3 * np.abs(L) * (Positions[JK[0], j] - Positions[i, j]) + (1 - r3) * (np.random.rand() > np.random.rand()) * (Positions[JK[2], j] - Positions[JK[1], j])

                Positions[i, :] = np.clip(Positions[i, :], lb, ub)

                SW_Fit1 = fobj(Positions[i, :])
                if SW_Fit1 < SW_Fit[i]:
                    SW_Fit[i] = SW_Fit1
                    if SW_Fit[i] < Best_score:
                        Best_score = SW_Fit[i]
                        Best_SW = Positions[i, :]
                else:
                    Positions[i, :] = O_P

                t += 1
                if t >= Tmax:
                    break

                Convergence_curve[t] = Best_score
        else:
            for i in range(SearchAgents_no):
                l = (a2 - 1) * np.random.rand() + 1
                SW_m = np.zeros(dim)
                O_P = Positions[i, :].copy()

                if SW_Fit[JK[0]] < SW_Fit[i]:
                    v1 = Positions[JK[0], :] - Positions[i, :]
                else:
                    v1 = Positions[i, :] - Positions[JK[0], :]

                if SW_Fit[JK[1]] < SW_Fit[JK[2]]:
                    v2 = Positions[JK[1], :] - Positions[JK[2], :]
                else:
                    v2 = Positions[JK[2], :] - Positions[JK[1], :]

                rn1 = np.random.randn()
                rn2 = np.random.randn()
                for j in range(dim):
                    SW_m[j] = Positions[i, j] + (np.exp(l)) * np.abs(rn1) * v1[j] + (1 - np.exp(l)) * np.abs(rn2) * v2[j]
                    if np.random.rand() < Cr:
                        Positions[i, j] = SW_m[j]

                Positions[i, :] = np.clip(Positions[i, :], lb, ub)

                SW_Fit1 = fobj(Positions[i, :])
                if SW_Fit1 < SW_Fit[i]:
                    SW_Fit[i] = SW_Fit1
                    if SW_Fit[i] < Best_score:
                        Best_score = SW_Fit[i]
                        Best_SW = Positions[i, :]
                else:
                    Positions[i, :] = O_P

                t += 1
                if t >= Tmax:
                    break

                Convergence_curve[t] = Best_score

        SearchAgents_no = int(N_min + (SearchAgents_no - N_min) * ((Tmax - t) / Tmax))

    return Best_score, Best_SW, Convergence_curve

def fobj(x):
    # rosenbrock 
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    # return np.sum(x ** 2)  # sphere

    # return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)   # rastrigin

    # ackley
    # a = 20
    # b = 0.2
    # c = 2 * np.pi
    # d = len(x)
    # sum1 = np.sum(x**2)
    # sum2 = np.sum(np.cos(c * x))
    # return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)


if __name__ == "__main__":
    SearchAgents_no = 30
    Tmax = 1000
    ub = 10 * np.ones(30)
    lb = -10 * np.ones(30)
    dim = 30

    Best_score, Best_SW, Convergence_curve = SWO(SearchAgents_no, Tmax, ub, lb, dim, fobj)

    print("Best Score:", Best_score)
    print("Best Solution:", Best_SW)
