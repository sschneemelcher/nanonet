#!/usr/bin/env python3

import numpy as np
from keras.datasets import mnist

from nanonet.model import get_model, predict, train, forward_pass
from nanonet.utils import one_hot



(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = np.max(y_train) + 1

x_train = (x_train / 255).reshape(x_train.shape[0], -1)
x_test = (x_test / 255).reshape(x_test.shape[0], -1)

y_test = y_test
y_train = one_hot(y_train)


layers = [
    {'input_shape': x_train.shape[1], 'units': 200, 'activation': 'relu'},
    {'units': 200, 'activation': 'relu'},
    {'units': 200, 'activation': 'relu'},
    {'units': 200, 'activation': 'relu'},
    {'units': num_classes, 'activation': 'softmax'},
]

model = get_model(layers)

perm = np.random.permutation(range(len(x_train)))
bs = 64
for e in range(50):
    for step in range(1, len(x_train) // bs):
        batch = perm[(step-1)*bs:step*bs]
        model = train(model, x_train[batch], y_train[batch], 0.01)
    preds = np.argmax(predict(model, x_test), axis=1)
    print(e+1, np.mean(preds == y_test))
