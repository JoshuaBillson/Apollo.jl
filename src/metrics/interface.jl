"""
Metrics are measures of a model's performance, such as loss, accuracy, or squared error.

Each metric must implement the following interface:
- `name(::Type{metric})`: Returns the human readable name of the metric.
- `init(metric)`: Returns the initial state of the metric as a `NamedTuple`.
- `update(metric, state, ŷ, y)`: Returns the new state given the previous state and a batch.
- `compute(metric, state)`: Computes the metric's value for the current state.

# Example Implementation
```julia
struct Accuracy <: ClassificationMetric end

name(::Type{Accuracy}) = "accuracy"

init(::Accuracy) = (correct=0, total=0)

function update(::Accuracy, state, ŷ, y)
    return (correct = state.correct + sum(ŷ .== y), total = state.total + length(ŷ))
end

compute(::Accuracy, state) = state.correct / max(state.total, 1)
```
"""
abstract type AbstractMetric end

"""
    name(m::AbstractMetric)
    name(m::Metric)

Human readable name of the given performance measure.
"""
name(::M) where {M <: AbstractMetric} = name(M)

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
    compute(metric::Metric)

Compute the performance measure from the current state.
"""
function compute end

"""
    Metric(measure::AbstractMetric)

`Metric` objects are used to store and maintain the state for a given `AbstractMetric`.

`Metrics` are compatible with the following interface:
- `name(metric)`: Returns the human readable name of the metric.
- `update!(metric, ŷ, y)`: Updates the metric's state for the given prediction/label pair.
- `reset!(metric)`: Restores the metric's state to the initial value.
- `compute(metric)`: Computes the metric's value for the current state.
"""
mutable struct Metric{M,T}
    measure::M
    state::T
end

Metric(measure::AbstractMetric) = Metric(measure, init(measure))

name(m::Metric) = name(m.measure)

"""
    update!(metric::Metric, ŷ, y)

Update the metric state for the next batch of labels and predictions.
"""
function update!(metric::Metric, ŷ, y)
    metric.state = update(metric.measure, metric.state, ŷ, y)
    return metric
end

compute(metric::Metric) = compute(metric.measure, metric.state)

"""
    reset!(metric::Metric)

Reset the metric's state.
"""
function reset!(metric::Metric)
    metric.state = init(metric.measure)
end

Base.show(io::IO, x::Metric) = print(io, "$(name(x)): $(round(compute(x), digits=4))")