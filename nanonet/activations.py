import numpy as np


leaky_slope = 10**-4

def relu(x):
    return np.maximum(x, 0)


def relu_(x):
    return x > 0


def leaky_relu(x):
    return np.maximum(0, x) + leaky_slope * np.minimum(0, x)


def leaky_relu_(x):
    return np.where(x > 0, 1, leaky_slope)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    s = np.sum(exp, axis=-1, keepdims=True)
    return exp / s


activation_map = {'relu': relu, 'sigmoid': sigmoid, 'lrelu': leaky_relu, 'softmax': softmax}
grad_map = {'relu': relu_, 'sigmoid': sigmoid_, 'lrelu': leaky_relu_}
