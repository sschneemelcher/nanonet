import numpy as np


def relu(x):
    return np.maximum(x, 0)


def relu_(x):
    return x > 0


def sigmoid(x):
    expp = np.exp(x)
    expn = np.exp(-x)
    return np.where(x >= 0, expp / (1 + expp), 1 / (1 + expn))


def sigmoid_(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    s = np.sum(exp, axis=-1, keepdims=True)
    return exp / s


activation_map = {'relu': relu, 'sigmoid': sigmoid, 'softmax': softmax}
grad_map = {'relu': relu_, 'sigmoid': sigmoid_}
