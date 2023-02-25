# 🧠 nanonet - a simple neural network library

`nanonet` is a lightweight neural network library written in Python, designed to provide a simple implementation for beginners who are interested in machine learning.

## 🌟 Features

`nanonet` provides the following features:

- 🏗️ Flexible model architecture definition using a simple syntax.
- 🚀 Commonly used activation functions such as ReLU, sigmoid, and softmax.
- 🔍 Support for backpropagation and stochastic gradient descent optimization.
- 🎉 Batch training for better model convergence.

## 🚀 Usage

Here's a simple example that demonstrates how to use `nanonet` to build a feedforward neural network:

```python
from keras.datasets import mnist

from nanonet.model import get_model, train, test
from nanonet.utils import one_hot


(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = max(y_train) + 1

x_train = (x_train / 255).reshape(x_train.shape[0], -1)
x_test = (x_test / 255).reshape(x_test.shape[0], -1)

y_train = one_hot(y_train)

# Describe the architecture
layers = [
    {'input_shape': x_train.shape[1], 'units': 200, 'activation': 'relu'},
    {'units': 16, 'activation': 'relu'},
    {'units': 16, 'activation': 'relu'},
    {'units': num_classes, 'activation': 'softmax'},
]

model = get_model(layers)

# Train the model
model = train(model, x_train, y_train, (x_test, y_test), bs=512, epochs=20, loss='mse')

# Evaluate the model
test(model, x_test, y_test)
```

## 🤝 Contributing

Contributions to nanonet are welcome! If you find a bug or have a feature request, please open an issue. If you'd like to contribute code, please fork the repository and submit a pull request.

## 📝 License

nanonet is released under the MIT License. See [LICENSE](./LICENSE) for details.