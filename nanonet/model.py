import numpy as np
from functools import reduce

from .activations import activation_map, grad_map


def get_model(layers):
    return [{'weights': [np.random.uniform(-.5, .5, (layers[i-1]['units'] if i > 0 else layers[i]['input_shape'], layers[i]['units'])),
                         np.random.uniform(-.5, .5, (1, layers[i]['units']))],
             'activation': layers[i]['activation']} for i in range(len(layers))]


def layer_reducer(acc, curr):
    logits = acc[-1]['activation'] @ curr['weights'][0] + curr['weights'][1]
    activation = activation_map[curr['activation']](logits)
    return acc + [{'logits': logits, 'activation': activation}]


def forward_pass(model, x):
    return reduce(layer_reducer, model, [{'activation': x, 'logits': np.ones(1)}])


def train(model, x, y, lr):
    m = len(x)
    f_pass = forward_pass(model, x)
    grads = []
    
    loss = f_pass[-1]['activation'] - y
    for i in reversed(range(0, len(model))):
        dw = f_pass[i]['activation'].T @ loss / m
        db = np.sum(loss, axis=0).reshape(1, -1) / m
        loss = loss @ model[i]['weights'][0].T * grad_map[model[i-1]['activation']](f_pass[i]['logits']) if i > 0 else 1
        grads.append([dw, db])
        
    return list(map(lambda update: {'weights': [update[0]['weights'][0] - lr * update[1][0], update[0]['weights'][1] - lr * update[1][1]], 'activation': update[0]['activation']}, zip(model, reversed(grads))))


def predict(model, x):
    return forward_pass(model, x)[-1]['activation']
