function update!(loss::AbstractLoss, model, opt_state, batch)
    x = batch[1:end-1]
    y = batch[end]
    grads = Flux.gradient(m -> loss(m(x...), y), model)
    Flux.update!(opt_state, model, grads[1])
end

function update!(loss::Union{<:WeightedLoss,<:MaskedLoss}, model, opt_state, batch)
    x = batch[1:end-2]
    y = batch[end-1]
    w = batch[end]
    grads = Flux.gradient(m -> loss(m(x...), y, w), model)
    Flux.update!(opt_state, model, grads[1])
end

function train!(loss::AbstractLoss, model, opt_state, data)
    foreach(batch -> update!(loss, model, opt_state, batch), data)
end