import numpy as np
import matplotlib.pyplot as plt

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
print(coefficients)
order = -coefficients[0]

print(f"Estimated order of convergence: {order:.4f}")

# Plot log-log
plt.figure(figsize=(8, 5))
plt.plot(log_steps, log_errors, 'o-', label='Error vs Step size')
plt.plot(log_steps, coefficients[1] + coefficients[0] * log_steps, 'r--',
         label=f'Fit: slope = {-order:.4f}')
plt.xlabel('log(N)')
plt.ylabel('log(Error)')
plt.title('Order of Convergence (Euler-Maruyama)')
plt.legend()
plt.grid()
plt.show()

module_tree_str = """
Wav2Vec2PreTrainedModel
Wav2Vec2ForPreTraining
  ├-Wav2Vec2GumbelVectorQuantizer
  └-Wav2Vec2Model
      ├-Wav2Vec2FeatureEncoder
      │   ├-Wav2Vec2NoLayerNormConvLayer
      │   ├-Wav2Vec2LayerNormConvLayer
      │   └-Wav2Vec2GroupNormConvLayer
      ├-Wav2Vec2FeatureProjection
      └-Wav2Vec2Encoder
          ├-Wav2Vec2PositionalConvEmbedding
          └-Wav2Vec2EncoderLayer
              ├-Wav2Vec2Attention
              └-Wav2Vec2FeedForward
"""