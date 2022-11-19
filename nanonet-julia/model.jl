using Base: tail

include("activations.jl")

function reduce_layer(acc, curr)
    # display(summary(get(last(acc), "weights", [])))
    input_shape = size(get(last(acc), "weights", [])[1])[2]
    output_shape = get(curr, "units", 1)
    push!(acc, Dict("weights" => [randn((input_shape, output_shape)), randn((1, output_shape))], "activation" => get(curr, "activation", "relu")))
end

function build_model(layers)
    input_shape = get(first(layers), "input_shape", 1)
    output_shape = get(first(layers), "units", 1)
    input_layer = Dict("weights" => [
            randn((input_shape, output_shape)), randn((1, output_shape))
        ], "activation" => "relu")
    reduce(reduce_layer, layers[2:length(layers)]; init=[input_layer])
end

reduce_predict(acc, curr) = acc * get(curr, "weights", [])[1] .+ get(curr, "weights", [])[2] |> get(activation_map, get(curr, "activation", "relu"), relu)

function reduce_keep_predict(acc, curr)
    logits = get(last(acc), "output", []) * get(curr, "weights", [])[1] .+ get(curr, "weights", [])[2]
    output = get(activation_map, get(curr, "activation", []), identity)(logits)
    return push!(acc, Dict("logits" => logits, "output" => output))
end


predict(model, x, keep_inters) = reduce(keep_inters ? reduce_keep_predict : reduce_predict, model; init=keep_inters ? [Dict("output" => x)] : x)
