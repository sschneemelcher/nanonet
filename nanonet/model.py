import numpy as np
from .activations import activation_map


def get_model(layers):
    model = []
    for i in range(len(layers)):
        layer = [{
            'weights': np.random.rand(layers[i-1]['units'] if i > 0 else layers[i]['input_shape'], layers[i]['units']) - 0.5,
            'activation': activation_map[layers[i]['activation']]
        }]
        model = model + layer
    return model


def predict(model, x):
    h = model[0]['activation'](x @ model[0]['weights'])
    for i in range(1, len(model)):
        h = model[i]['activation'](h @ model[i]['weights'])
    return h
