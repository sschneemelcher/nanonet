relu(x::Matrix{Float64}) = x .* (x .> 0)

softmax(x::Matrix{Float64}) = ℯ .^ (x .- maximum(x, dims=2)) ./ sum(ℯ .^ (x .- maximum(x, dims=2)), dims=2)

activation_map = Dict("relu" => relu, "softmax" => softmax)
