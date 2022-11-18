using MLDatasets

relu(x::Matrix{Float64}) = x .* (x .> 0)

softmax(x::Matrix{Float64}) = ℯ .^ (x .- maximum(x, dims=2)) ./ sum(ℯ .^ (x .- maximum(x, dims=2)), dims=2)

x_train, y_train = MNIST.traindata(Float64, 1:10)
x_train = reshape(permutedims(x_train, [3, 1, 2]), (length(x_train[1, 1, :]), 784))

model = [[randn((784, 16)), relu],
    [randn((16, 16)), relu],
    [randn((16, 2)), softmax]]

f(acc, curr) = acc * curr[1] |> curr[2]
reduce(f, model; init=x_train) |> display
