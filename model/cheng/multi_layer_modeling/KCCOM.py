import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# 定义KCCMO算法
def KCCMO(objective_function, helper_function, bounds, num_iterations=100, population_size=50, surrogate_threshold=0.5):
    bounds = np.array(bounds)
    Population1 = np.random.uniform(
        bounds[:, 0], bounds[:, 1], (population_size, len(bounds)))
    Population2 = np.random.uniform(
        bounds[:, 0], bounds[:, 1], (population_size, len(bounds)))

    Eval1 = np.array([objective_function(ind) for ind in Population1])
    Eval2 = np.array([helper_function(ind) for ind in Population2])

    # 初始化克里金模型
    kernel = Matern(length_scale=1.0, nu=2.5)
    surrogate_model = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10)

    for iteration in range(num_iterations):
        Parent1 = Population1[:population_size // 2]
        Parent2 = Population2[:population_size // 2]

        Off1 = Parent1 + np.random.normal(0, 0.1, Parent1.shape)
        Off2 = Parent2 + np.random.normal(0, 0.1, Parent2.shape)

        # 自适应切换策略
        if iteration / num_iterations < surrogate_threshold:
            # 使用克里金模型进行替代优化
            surrogate_model.fit(Population1, Eval1[:, 0])
            surrogate_eval = surrogate_model.predict(np.vstack((Off1, Off2)))
            Eval1 = np.hstack((surrogate_eval.reshape(-1, 1),
                              Eval1[:, 1].reshape(-1, 1)))
        else:
            # 使用原始目标函数进行优化
            Eval1 = np.array([objective_function(ind)
                             for ind in np.vstack((Off1, Off2))])

        Eval2 = np.array([helper_function(ind)
                         for ind in np.vstack((Off1, Off2))])

        Population1 = np.vstack((Population1, Off1, Off2))
        Population2 = np.vstack((Population2, Off1, Off2))

        # 数据选择策略和修改的填充采样准则
        idx1 = np.lexsort(Eval1.T)
        idx2 = np.argsort(Eval2)
        Population1 = Population1[idx1][:population_size]
        Population2 = Population2[idx2][:population_size]

    return Population1[0], Eval1[0]
