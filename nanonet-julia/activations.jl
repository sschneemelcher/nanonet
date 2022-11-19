struct Activation
    forward::Function
    backward::Function
end


relu(x::Matrix{Float64}) = x .* (x .> 0)

relu_(x::Matrix{Float64}) = x .> 0

Relu = Activation(relu, relu_)

softmax(x::Matrix{Float64}) = ℯ .^ (x .- maximum(x, dims=2)) ./ sum(ℯ .^ (x .- maximum(x, dims=2)), dims=2)

Softmax = Activation(softmax, identity)

sigmoid(x) =  1 ./ (1 .+ (ℯ .^ -x))

sigmoid_(x) = sigmoid(x) .* (1.0 - sigmoid(x))

Sigmoid = Activation(sigmoid, sigmoid_) 

None = Activation(identity, identity)
