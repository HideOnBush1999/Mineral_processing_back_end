import numpy as np

def ZOA(func, lb, ub, dim, N, T):
    # Initialize the population of zebras
    X = np.random.uniform(lb, ub, (N, dim))
    # Evaluate the objective function for each zebra
    F = np.array([func(x) for x in X])
    # Find the index of the best zebra (pioneer zebra)
    PZ_index = np.argmin(F)
    PZ = X[PZ_index]
    # Initialize convergence curve
    Convergence_curve = np.zeros(T)
    for t in range(T):
        for i in range(N):
            # Phase 1: Foraging Behavior
            r = np.random.rand()
            I = np.round(1 + np.random.rand())
            X_new = X[i] + r * (PZ - I * X[i])
            F_new = func(X_new)
            if F_new < F[i]:
                X[i] = X_new
                F[i] = F_new
            # Update the pioneer zebra if needed
            if F_new < F[PZ_index]:
                PZ_index = i
                PZ = X[i]
            # Phase 2: Defense Strategies Against Predators
            Ps = np.random.rand()
            if Ps <= 0.5:
                # Strategy 1: Escape from lion
                R = 0.01
                X_new = X[i] + R * (2 * r - 1)
            else:
                # Strategy 2: Defend against other predators
                AZ_index = np.random.randint(0, N)  # Randomly select an attacked zebra
                AZ = X[AZ_index]
                X_new = X[i] + r * (AZ - I * X[i])
            F_new = func(X_new)
            if F_new < F[i]:
                X[i] = X_new
                F[i] = F_new
            # Update the pioneer zebra if needed
            if F_new < F[PZ_index]:
                PZ_index = i
                PZ = X[i]
        # Record the best value at each iteration
        Convergence_curve[t] = F[PZ_index]
    # Return the best solution found and the convergence curve
    best_index = np.argmin(F)
    return X[best_index], F[best_index], Convergence_curve


# Example usage
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

if __name__ == '__main__':
    lb = -5.12
    ub = 5.12
    dim = 30
    N = 50
    T = 1000

    best_pos, best_val, Convergence_curve = ZOA(fobj, lb, ub, dim, N, T)
    print("Best position:", best_pos)
    print("Best value:", best_val)
