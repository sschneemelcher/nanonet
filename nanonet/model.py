import numpy as np
from functools import reduce

from .activations import activation_map


def get_model(layers):
    return [{'weights': [np.random.uniform(-.5, .5, (layers[i-1]['units'] if i > 0 else layers[i]['input_shape'], layers[i]['units'])),
                         np.random.uniform(-.5, .5, (1, layers[i]['units']))],
             'activation': layers[i]['activation']} for i in range(len(layers))]


def predict(model, x):
    return reduce(lambda acc, curr: activation_map[curr['activation']](acc @ curr['weights'][0] + curr['weights'][1]), model, x)


def train(model, x, y, loss_function):
    pass
