import numpy as np


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2, axis=-1, keepdims=True)


def mse_(y_pred, y_true):
    return 2 * (y_pred - y_true)


def mae(y_pred, y_true):
    return np.mean(np.abs((y_pred - y_true)), axis=-1, keepdims=True)


def mae_(y_pred, y_true):
    return np.where(y_pred > y_true, 1., -1.)


loss_map = {'mse': mse, 'mae': mae}
loss_grad_map = {'mse': mse_, 'mae': mae_}
