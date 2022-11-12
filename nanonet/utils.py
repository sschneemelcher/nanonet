import numpy as np


def normalize(x):
    min = np.min(x, axis=-1, keepdims=True)
    return 2 * (x - np.max(x, axis=-1, keepdims=True) - min) - 1


def one_hot(y):
    y_new = np.zeros((len(y), max(y) + 1))
    y_new[np.arange(len(y)), y] = 1
    return y_new
