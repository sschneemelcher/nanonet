include("activations.jl")

struct Layer
    weights::Matrix{Float64}
    bias::Matrix{Float64}
    shape::Tuple{Int64, Int64}
    activation_f::Activation
end

function reduce_layer(acc, curr)
    input_shape = last(acc).shape[2]
    output_shape = get(curr, "units", 1)
    return push!(acc, Layer(randn((input_shape, output_shape)), randn((1, output_shape)), (input_shape, output_shape), get(curr, "activation", None)))
end

function build_model(layers)
    input_shape = get(first(layers), "input_shape", 1)
    output_shape = get(first(layers), "units", 1)
    input_layer = Layer(randn((input_shape, output_shape)), randn((1, output_shape)), (input_shape, output_shape), get(first(layers), "activation", None))
    return reduce(reduce_layer, layers[2:length(layers)]; init=[input_layer])
end

function reduce_keep_predict(acc, curr)
    logits = get(last(acc), "output", []) * curr.weights .+ curr.bias
    output = curr.activation_f.forward(logits)
    return push!(acc, Dict("logits" => logits, "output" => output))
end

reduce_predict(acc, curr) = acc * curr.weights .+ curr.bias |> curr.activation_f.forward

predict(model, x, keep_inters=false) = reduce(keep_inters ? reduce_keep_predict : reduce_predict, model; init=keep_inters ? [Dict("output" => x)] : x)
