import numpy as np
from nanonet.model import get_model, predict

layers = [
    {'input_shape': 8, 'units': 16, 'activation': 'relu'},
    {'units': 16, 'activation': 'relu'},
    {'units': 2, 'activation': 'softmax'},
]

model = get_model(layers)


X = np.random.random((1, 8))

print(predict(model, X))
