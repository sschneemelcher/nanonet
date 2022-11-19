function one_hot(y) 
    y_new = zeros((size(y)[1], maximum(y) + 1)) 
    [y_new[a, b] = 1 for (a, b) in zip(collect(1:size(y)[1]), y[:] .+ 1)]
    return y_new
end
