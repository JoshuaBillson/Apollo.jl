function fit(model, traindata, valdata, opt, loss; gpu=false, epochs=1)
    # Copy Model
    model = gpu ? model |> Flux.gpu : deepcopy(model)

    # Initialize Optimiser
    opt_state = Flux.setup(opt, model)

    # Store Best Model
    fitresult = gpu ? model |> Flux.cpu : model

    # Iterate Over Epochs
    for epoch in 1:epochs

        # Iterate Over Train Data
        for (x, y) in traindata

            # Update Model
            grad = Flux.gradient(m -> loss(m(x), y), model)
            Flux.update!(opt_state, model, grad[1])

        end

        # Evaluate Model

    end
end