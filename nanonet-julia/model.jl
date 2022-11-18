include("activations.jl")

function build_model(layers)
    layers = map(Dict, layers)
    model = []
    for i in collect(1:length(layers))
        layer = [randn((haskey(layers[i], "input_shape")
                        ? get(layers[i], "input_shape", 16)
                        : get(layers[i-1], "units", 16), get(layers[i], "units", 16))), get(layers[i], "activation", "relu")]
        model = push!(model, layer)
    end
    return model
end

predict(acc, curr) = acc * curr[1] |> get(activation_map, curr[2], identity)
