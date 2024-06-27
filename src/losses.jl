function binarycrossentropy(ŷ, y, w=MLUtils.ones_like(y))
    l = Flux.binarycrossentropy(ŷ, y, agg=identity)
    return mean(l .* w)
end