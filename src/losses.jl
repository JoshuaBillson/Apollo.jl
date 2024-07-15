function binarycrossentropy(ŷ, y, w=ones_like(y))
    l = Flux.binarycrossentropy(ŷ, y, agg=identity)
    return mean(l .* w)
end

function mae(ŷ, y, w=ones_like(y))
    return mean(abs.(ŷ .- y) .* w)
end

function mse(ŷ, y, w=ones_like(y))
    return mean(((ŷ .- y) .^ 2) .* w)
end