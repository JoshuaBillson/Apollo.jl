"""
Abstract supertype of all evaluation metrics.
"""
abstract type AbstractMetric end

"""
    name(m::AbstractMetric)
    name(m::Metric)

Human readable name of the given performance measure.
"""
function name end

"""
    init(m::AbstractMetric)

Returns the initial state of the performance measure, which will be subsequently updated
for each mini-batch of labels and predictions.
"""
function init end

"""
    update(m::AbstractMetric, state, ŷ, y)

Return the new state for the given batch of labels and predictions.
"""
function update end

"""
    compute(m::AbstractMetric, state)
    compute(m::Metric, state)

Compute the performance measure from the current state.
"""
function compute end

"""
    Metric(measure::AbstractMetric)

Construct a Metric object to track the state for the given `AbstractMetric`.
"""
mutable struct Metric{M,T}
    measure::M
    state::T
end

Metric(measure::AbstractMetric) = Metric(measure, init(measure))

function name(m::Metric{M}) where {M <: AbstractMetric}
    return name(m.measure)
end

"""
    update!(metric::Metric, ŷ, y)

Update the metric state for the next batch of labels and predictions.
"""
function update!(metric::Metric, ŷ, y)
    metric.state = update(metric.measure, metric.state, ŷ, y)
    return metric
end

function compute(metric::Metric)
    return compute(metric.measure, metric.state)
end

Base.show(io::IO, x::Metric) = print(io, "$(name(x)): $(round(compute(x), digits=4))")

"""
    evaluate(model, data, measures...)
    evaluate(model::BinarySegmentationModel, data, measures...)

Evaluate the model's performance on the provided data.

# Parameters
- `model`: A callable that takes a single batch from `data` and returns a tuple of the form (ŷ, y).
- `data`: An iterable of (x, y) values.
- `measures`: A set of `AbstractMetrics` to use for evaluating `model`.

# Returns
A `NamedTuple` containing the performance metrics for the given model.

# Example
```julia
evaluate(DataLoader((xsampler, ysampler)), Accuracy(), MIoU(2)) do (x, y)
    ŷ = model(x) |> Flux.sigmoid
    return (ŷ, y)
end
```
"""
function evaluate(model, data, measures::Vararg{AbstractMetric})
    metrics = map(Metric, measures)
    for batch in data
        ŷ, y = model(batch)
        foreach(metric -> update!(metric, ŷ, y), metrics)
    end
    vals = map(compute, metrics)
    names = map(Symbol ∘ name, metrics)
    return NamedTuple{names}(vals)
end

evaluate(model::BinarySegmentationModel, data) = evaluate(model, data, Accuracy(), MIoU(2))
function evaluate(model::BinarySegmentationModel, data, metrics::Vararg{AbstractMetric})
    evaluate(data, metrics...) do batch
        x = batch[1:end-1]
        y = batch[end]
        return map(Flux.cpu, (model(x...), y))
    end
end