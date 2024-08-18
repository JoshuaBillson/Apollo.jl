function update!(loss::AbstractLoss, model, opt_state, batch)
    grads = Flux.gradient(m -> loss(m, batch), model)
    Flux.update!(opt_state, model, grads[1])
end

function train!(loss::AbstractLoss, model, opt_state, data)
    foreach(batch -> update!(loss, model, opt_state, batch), data)
end