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

#function build_model(layers)
#    layers = map(Dict, layers)
#    model = []
#    for i in collect(1:length(layers))
#        layer = [randn((haskey(layers[i], "input_shape")
#                        ? get(layers[i], "input_shape", 16)
#                        : get(layers[i-1], "units", 16), get(layers[i], "units", 16))), get(layers[i], "activation", "relu")]
#        model = push!(model, layer)
#    end
#    return model
#end

f(acc, curr) = acc * get(curr, "weights", [])[1] .+ get(curr, "weights", [])[2] |> get(activation_map, get(curr, "activation", "relu"), relu)
predict(model, x) = reduce(f, model; init=x)
