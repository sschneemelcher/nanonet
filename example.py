#!/usr/bin/env python3

import numpy as np
from nanonet.model import get_model, predict

layers = [
    {'input_shape': 4, 'units': 8, 'activation': 'relu'},
    {'units': 8, 'activation': 'relu'},
    {'units': 8, 'activation': 'relu'},
    {'units': 2, 'activation': 'softmax'},
]

model = get_model(layers)

print(model)

X = np.random.random((1, 4))

print(predict(model, X))
