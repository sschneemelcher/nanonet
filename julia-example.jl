#!/usr/bin/env julia

include("nanonet-julia/model.jl")
include("nanonet-julia/utils.jl")


using MLDatasets

trainset = MNIST(:train)

x_train, y_train = trainset.features, trainset.targets
x_train = reshape(permutedims(x_train, [3, 1, 2]), (length(x_train[1, 1, :]), length(x_train[:, :, 1])))

num_classes = 10

layers = [
    Dict("input_shape" => length(x_train[1, :]), "units" => 200, "activation" => Relu),
    Dict("units" => 16, "activation" => Sigmoid),
    Dict("units" => 16, "activation" => Relu),
    Dict("units" => num_classes, "activation" => Softmax)
]

model = build_model(layers)

predict(model, x_train) |> display
