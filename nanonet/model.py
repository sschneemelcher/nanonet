import numpy as np
from .activations import activation_map


def get_model(layers):
    return [{'weights': [np.random.rand(layers[i-1]['units'] if i > 0 else layers[i]['input_shape'], layers[i]['units']) - 0.5,
                         np.random.rand(1, layers[i]['units'])],
             'activation': layers[i]['activation']} for i in range(len(layers))]


def predict(model, x):
    h = activation_map[model[0]['activation']](x @ model[0]['weights'][0] + model[0]['weights'][1])
    for i in range(1, len(model)):
        h = activation_map[model[i]['activation']](h @ model[i]['weights'][0] + model[i]['weights'][1])
    return h


def train(model, x, y, loss_function):
    pass
