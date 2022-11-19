#!/usr/bin/env julia

include("nanonet-julia/model.jl")

using MLDatasets

trainset = MNIST(:train)

x_train, y_train = trainset.features, trainset.targets
x_train = reshape(permutedims(x_train, [3, 1, 2]), (length(x_train[1, 1, :]), length(x_train[:, :, 1])))

num_classes = 10

layers = [
    Dict("input_shape" => length(x_train[1, :]), "units" => 200, "activation" => "relu"),
    Dict("units" => 16, "activation" => "relu"),
    Dict("units" => 16, "activation" => "relu"),
    Dict("units" => num_classes, "activation" => "softmax")
]

model = build_model(layers)

predict(model, x_train) |> display
