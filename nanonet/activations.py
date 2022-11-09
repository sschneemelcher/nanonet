import numpy as np


def relu(x):
    return np.maximum(x, 0)


def relu_(x):
    return np.where(x > 0, x, 0)


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


activation_map = {'relu': relu, 'softmax': softmax}
grad_map = {'relu': relu_}
