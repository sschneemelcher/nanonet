#!/usr/bin/env julia

include("nanonet-julia/model.jl")

using MLDatasets

x_train, y_train = MNIST.traindata(Float64, 1:10)
x_train = reshape(permutedims(x_train, [3, 1, 2]), (length(x_train[1, 1, :]), 784))

num_classes = 10

layers = ((("input_shape", length(x_train[1, :])), ("units", 200), ("activation", "relu")),
    (("units", 16), ("activation", "relu")),
    (("units", 16), ("activation", "relu")),
    (("units", num_classes), ("activation", "softmax")))

model = build_model(layers)

reduce(predict, model; init=x_train) |> display
