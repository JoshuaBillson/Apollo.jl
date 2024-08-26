"""
    evaluate(model, data, metrics...; gpu=false)

Evaluate the model's performance on the provided data.

# Parameters
- `model`: A model to be evaluated, where `ŷ = model(xs...)`.
- `data`: An iterable of batches with the form `(xs..., y)`.
- `metrics`: A set of `AbstractMetrics` with which to evaluate `model`.
- `gpu`: If true, features will be moved to the gpu before being passed to `model`.

# Returns
A `NamedTuple` with the performance metrics for the given model.

# Example
```julia
julia> evaluate(model, traindata, Accuracy(), MIoU([1,2]), gpu=true)
(accuracy = 0.9665, MIoU = 0.9204)
```
"""
function evaluate(model, data, measures::Vararg{AbstractMetric}; gpu=false)
    return @pipe MetricLogger(measures...) |> evaluate!(model, data, _; gpu=gpu) |> scores
end

function evaluate!(model, data, logger::Union{<:Tracker,<:MetricLogger}; gpu=false, metrics=r".*")
    for (xs..., y) in data
        ŷ = @match gpu begin
            true => @pipe map(Flux.gpu, xs) |> model(_...) |> Flux.cpu
            false => model(xs...)
        end
        step!(logger, metrics, ŷ, y)
    end
    return logger
end