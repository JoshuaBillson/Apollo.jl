function fit(
	task::BinarySegmentation, train, val;
	opt=Flux.Optimisers.Adam(), 
	loss=Flux.binarycrossentropy, 
	metrics=[Apollo.MIoU([0,1]), Apollo.Accuracy()], 
	gpu=false, 
	epochs=10, 
	freeze_encoder=true
    )

    # Initialize Training Params
    params = (gpu=gpu, opt=opt, loss=loss, metrics=metrics, epochs=epochs, freeze_encoder=freeze_encoder)

    # Initialize Training State
    state = init_state(task, params)

    # Initialize Data
    data = gpu ? (train=Flux.gpu(train), val=Flux.gpu(val)) : (train=train, val=val)

    # Iterate Over Epochs
    for epoch in 1:epochs
        state = fit_one_cycle(task, data, params, state)
    end

    # Return Trained Model
    return BinarySegmentation(state.fitresult)
end

function init_state(x::BinarySegmentation, params)
    model = params.gpu ? x.model |> Flux.gpu : deepcopy(x.model)
    fitresult = params.gpu ? model |> Flux.cpu : deepcopy(model)
    opt_state = Flux.setup(params.opt, model)
    params.freeze_encoder && _freeze_encoder!(model, opt_state)
	tracker = _build_tracker(Apollo.Loss(params.loss), params.metrics...)
    return (model=model, fitresult=fitresult, opt_state=opt_state, tracker=tracker)
end

function fit_one_cycle(::BinarySegmentation, data, params, state)
    # Update Model
    for (x, y) in data.train
        grads = Flux.gradient(m -> params.loss(Flux.sigmoid(m(x)), y), state.model)
        Flux.update!(state.opt_state, state.model, grads[1])
    end

    # Evaluate Train Performance
    for (x, y) in data.train
        Apollo.step!(state.tracker, r"train_", Flux.sigmoid(state.model(x)), y)
    end

    # Evaluate Val Performance
    for (x, y) in data.val
        Apollo.step!(state.tracker, r"val_", Flux.sigmoid(state.model(x)), y)
    end

    # End Epoch
    @info Apollo.printscores(state.tracker, metrics=r"_loss")
    Apollo.epoch!(state.tracker)

    # Update Fitresult
    if best_epoch(state.tracker, Apollo.Min("val_loss")) == (current_epoch(state.tracker) - 1)
        fitresult = params.gpu ? state.model |> Flux.cpu : deepcopy(state.model)
        state = @set state.fitresult = fitresult
    end

    # Return New State
    return state
end

evaluate(task::BinarySegmentation, data) = evaluate(task, data, Accuracy(), MIoU(2))
function evaluate(task::BinarySegmentation, data, metrics::Vararg{AbstractMetric})
    evaluate(data, metrics...) do batch
        x = batch[1:end-1]
        y = batch[end]
        return map(Flux.cpu, (task(x...), y))
    end
end

function _build_tracker(metrics...)
	train_metrics = ["train_$(Apollo.name(m))" => m for m in metrics]
	val_metrics = ["val_$(Apollo.name(m))" => m for m in metrics]
	return Apollo.Tracker(train_metrics..., val_metrics...)
end

_freeze_encoder!(::UNet, opt_state) = Flux.Optimisers.freeze!(opt_state.encoder)
_freeze_encoder!(::DeeplabV3, opt_state) = Flux.Optimisers.freeze!(opt_state.encoder)
