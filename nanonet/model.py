import numpy as np
from functools import reduce
from json import load, dump

from .activations import activation_map, grad_map, sigmoid_
from .losses import loss_grad_map, loss_map
from .utils import normalize


def get_model(layers):
    return [{'weights': [np.random.uniform(-.5, .5, (layers[i-1]['units'] if i > 0 else layers[i]['input_shape'], layers[i]['units'])),
                         np.random.uniform(-.5, .5, (1, layers[i]['units']))],
             'activation': layers[i]['activation']} for i in range(len(layers))]


def layer_reducer(acc, curr):
    logits = acc[-1]['activation'] @ curr['weights'][0] + curr['weights'][1]
    activation = activation_map[curr['activation']](logits)
    return acc + [{'logits': logits, 'activation': activation}]


def predict(model, x, keep_intermediates=False):
    if keep_intermediates:
        return reduce(layer_reducer, model, [{'activation': x, 'logits': np.ones(1)}])
    else:
        return reduce(lambda acc, curr: activation_map[curr['activation']](acc @ curr['weights'][0] + curr['weights'][1]), model, x)


def lfunc(z, model, f_pass, m):
    dw = f_pass[-1]['activation'].T @ z 
    db = np.sum(z, axis=0).reshape(1, -1) 
    da = grad_map[model[-2]['activation']](f_pass[-1]['activation']) if len(model) > 1 else 1    

    if len(model) == 1:
        z = z @ model[-1]['weights'][0].T
        return [[dw, db], z * grad_map[model[0]['activation']](f_pass[0]['activation'])]

    return [[dw, db]] + lfunc(z @ model[-1]['weights'][0].T * da, model[:-1], f_pass[:-1], m)


def get_grads(model, x, y, loss):
    # make sure that labels shape fits output layer
    assert(y.shape[-1] == model[-1]['weights'][0].shape[-1])
    assert(len(x.shape) > 1)
    assert(loss_grad_map[loss])

    f_pass = predict(model, x, keep_intermediates=True)
    z = np.mean(loss_grad_map[loss](f_pass[-1]['activation'], y), axis=0, keepdims=True)
    f_pass = list(map(lambda x: {'logits': np.mean(x['logits'], axis=0, keepdims=True), 'activation': np.mean(x['activation'], axis=0, keepdims=True) },f_pass))
    # list(map(lambda x: print(x['activation'].shape), f_pass))

    return lfunc(z, model, f_pass[:-1], x.shape[0])


def update_model(model, grads, lr):
    return list(map(lambda update: {'weights': [update[0]['weights'][0] - lr * update[1][0], update[0]['weights'][1] - lr * update[1][1]], 'activation': update[0]['activation']}, zip(model, reversed(grads))))


def train(model, x, y, val_data=None, bs=64, epochs=10, lr=0.1, loss='mse'):
    assert(len(x.shape) > 1)

    if val_data is not None:
        x_val, y_val = val_data
        if len(y_val.shape) > 1:
            y_val = np.argmax(
                y_val, axis=-1) if y_val.shape[-1] > 1 else y_val.reshape(-1)

    perm = np.random.permutation(range(x.shape[0]))
    bs = 512
    for e in range(epochs):
        for step in range(1, len(x) // bs):
            batch = perm[(step-1)*bs:step*bs]
            grads = get_grads(model, x[batch], y[batch], loss)
            model = update_model(model, grads[:-1], lr)

        if val_data is not None:
            preds = np.argmax(predict(model, x_val), axis=1)
            print(e+1, np.mean(preds == y_val))

    return model


def test(model, x, y):
    assert(len(x.shape) > 1)
    if len(y.shape) > 1:
        y = np.argmax(y, axis=-1) if y.shape[-1] > 1 else y.reshape(-1)

    preds = np.argmax(predict(model, x), axis=1)
    print(f'mean: {np.mean(preds == y)}')


def save_model(model, path):
    s_model = []
    for layer in model:
        s_model.append({'weights': [layer['weights'][0].tolist(
        ), layer['weights'][1].tolist()], 'activation': layer['activation']})

    with open(path, 'w') as f:
        dump(s_model, f)


def load_model(path):
    with open(path, 'r') as f:
        s_model = load(f)

    model = []
    for layer in s_model:
        model.append({'weights': [np.asarray(layer['weights'][0]), np.asarray(
            layer['weights'][1])], 'activation': layer['activation']})

    return model
