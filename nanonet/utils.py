import numpy as np


def one_hot(y):
    y_new = np.zeros((len(y), max(y) + 1))
    y_new[np.arange(len(y)), y] = 1
    return y_new
