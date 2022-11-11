import numpy as np


def relu(x):
    return np.maximum(x, 0)


def relu_(x):
    return x > 0


def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    s = np.sum(exp, axis=-1, keepdims=True)
    return exp / s


activation_map = {'relu': relu, 'softmax': softmax}
grad_map = {'relu': relu_}
