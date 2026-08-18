"""DATA110 Lab 04 - ANN fundamentals and weight update practice."""
import numpy as np

# Instructor's core update equation:
# W_new = W_old - learning_rate * dLoss/dW

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Simple numerical example
w_old = 0.50
learning_rate = 0.10
gradient = 0.20
w_new = w_old - learning_rate * gradient
print("Updated weight:", w_new)

# Batch styles to remember:
# full batch = update after all observations
# mini-batch = update after a small group
# online = update after one observation

# Vanishing gradient: repeated multiplication of very small gradients can make
# the final gradient close to zero, so weights barely change in deep networks.
