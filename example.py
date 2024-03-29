#!/usr/bin/env python3

from keras.datasets import mnist

from nanonet.model import get_model, predict, train, save_model, load_model, test
from nanonet.utils import one_hot


(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = max(y_train) + 1

x_train = (x_train / 255).reshape(x_train.shape[0], -1)
x_test = (x_test / 255).reshape(x_test.shape[0], -1)

y_train = one_hot(y_train)


layers = [
    {'input_shape': x_train.shape[1], 'units': 200, 'activation': 'relu'},
    {'units': 16, 'activation': 'relu'},
    {'units': 16, 'activation': 'relu'},
    {'units': num_classes, 'activation': 'softmax'},
]

model = get_model(layers)
model = train(model, x_train, y_train, (x_test, y_test), bs=512, epochs=20, loss='mse')

# model_name = 'mynet.json'
# save_model(model, model_name)
# model = load_model(model_name)
# test(model, x_train, y_train)
