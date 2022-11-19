function one_hot(y) 
    y_new = zeros((length(y), maximum(y) + 1)) 
    [y_new[a, b] = 1 for (a, b) in zip(1:length(y), y .+ 1)]
    return y_new
end
