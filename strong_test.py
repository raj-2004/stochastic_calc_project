import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def perform_test():
    # Parameters
    X0 = 1.0  # Initial value
    T = 1  # Final time
    M = 1000  # Number of paths for Monte Carlo
    steps_list = [i for i in range(100, 5000, 100)]  # Different step sizes: [16, 32, ..., 1024]


    # Closed-form solution for X_t
    def exact_solution(X0, T, W_T):
        return (np.sqrt(X0) + W_T - T / 2) ** 2


    # Euler-Maruyama method
    def euler_maruyama(X0, T, N, M):
        dt = T / N
        X_approx = np.zeros((M, N + 1))
        X_approx[:, 0] = X0

        dW = np.random.normal(0, np.sqrt(dt), (M, N))

        for i in range(1, N + 1):
            X_prev = X_approx[:, i - 1]
            X_prev[X_prev < 0] = 0  # Ensure sqrt(X_prev) is real
            X_approx[:, i] = X_prev + dt + 2 * np.sqrt(X_prev) * dW[:, i - 1]

        return X_approx[:, -1]  # Return X_T for all paths


    # Compute errors for different step sizes
    errors = []
    for N in steps_list:
        # Simulate exact solution
        W_T = np.random.normal(0, np.sqrt(T), M)
        X_true = exact_solution(X0, T, W_T)

        # Simulate approximation
        X_approx = euler_maruyama(X0, T, N, M)

        # Compute mean absolute error
        error = np.mean(np.abs(X_true - X_approx))
        errors.append(error)

    # Fit a line to log-log data to find the order of convergence
    log_steps = np.log(np.array(steps_list))
    log_errors = np.log(np.array(errors))
    coefficients = np.polyfit(log_steps, log_errors, 1)
    order = -coefficients[0]
    return order

all_test_results = []
for i in tqdm(range(100)):
    all_test_results.append(perform_test())
all_test_results = np.array(all_test_results)
final_order = np.mean(all_test_results)