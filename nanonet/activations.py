import numpy as np


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


activation_map = {'relu': relu, 'softmax': softmax}
