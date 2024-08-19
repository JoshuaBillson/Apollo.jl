abstract type Order end

"""
    Max(metric::String)

Represents an ordering of largest to smallest for the given metric.
"""
struct Max <: Order 
    metric::String
end

"""
    Min(metric::String)

Represents an ordering of smallest to largest for the given metric.
"""
struct Min <: Order 
    metric::String
end

"""
    MetricLogger(metrics...; prefix="")

An object to track one or more metrics. Each metric is associated with a unique name, 
which defaults to `name(metric)`. This can be overriden by providing a `name => metric`
pair. The `prefix` keyword adds a constant prefix string to every name.

# Example 1
```jldoctest
julia> md = MetricLogger(Accuracy(), MIoU([0,1]); prefix="train_");

julia> step!(md, [0, 0, 1, 0], [0, 0, 1, 1]);

julia> md
MetricLogger(train_accuracy=0.75, train_MIoU=0.5833333333333334)

julia> step!(md, [0, 0, 1, 1], [0, 0, 1, 1]);

julia> md
MetricLogger(train_accuracy=0.875, train_MIoU=0.775)

julia> reset!(md)

julia> md
MetricLogger(train_accuracy=0.0, train_MIoU=1.0)
```

# Example 2
```jldoctest
julia> md = MetricLogger("train_acc" => Accuracy(), "val_acc" => Accuracy());

julia> step!(md, "train_acc", [0, 1, 1, 0], [1, 1, 1, 0]);  # update train_acc

julia> step!(md, r"val_", [1, 1, 1, 0], [1, 1, 1, 0]);  # update metrics matching regex

julia> md
MetricLogger(train_acc=0.75, val_acc=1.0)
```
"""
struct MetricLogger{D<:OrderedDict}
    metrics::D
end

function MetricLogger(metrics...; prefix="")
    # Preprocess Metric Names
    metrics = @pipe map(_preprocess_metric, metrics) |> [(prefix * n) => m for (n, m) in _]

    # Populate Metric Dict
    metric_dict = OrderedDict{String,Any}()
    for (name, metric) in metrics
        metric_dict[name] = Metric(metric)
    end

    # Return MetricLogger
    return MetricLogger(metric_dict)
end

_preprocess_metric(m::Pair{String, <:AbstractMetric}) = m
_preprocess_metric(m::Pair{:Symbol, <:AbstractMetric}) = string(first(m)) => last(m)
_preprocess_metric(m::AbstractMetric) = name(m) => m

Base.keys(x::MetricLogger) = keys(x.metrics)
Base.getindex(x::MetricLogger, i) = x.metrics[i]
Base.setindex!(x::MetricLogger, val, key...) = Base.setindex!(x.metrics, val, keys...)
Base.pairs(x::MetricLogger) = Base.pairs(x.metrics)

function Base.show(io::IO, ::MIME"text/plain", x::MetricLogger)
    printstyled(io, "MetricLogger(")
    for (i, (name, metric)) in enumerate(x.metrics)
        printstyled(io, "$name")
        printstyled(io, "=")
        printstyled(io, "$(compute(metric))")
        i < length(x.metrics) && printstyled(io, ", ")
    end
    printstyled(io, ")")
end

"""
    step!(x::MetricLogger, ŷ, y)
    step!(x::MetricLogger, metric::String, ŷ, y)
    step!(x::MetricLogger, metric::Regex, ŷ, y)

Update the metric for the current epoch using the provided prediction/label pair.
"""
step!(md::MetricLogger, ŷ, y) = step!(md, r".*", ŷ, y)
step!(md::MetricLogger, metric::String, ŷ, y) = update!(md.metrics[metric], ŷ, y)
function step!(md::MetricLogger, pattern::Regex, ŷ, y)
    foreach(metric -> step!(md, metric, ŷ, y), _filter_metrics(md, pattern))
end

_filter_metrics(md::MetricLogger, pat::Regex) = filter(x -> contains(x, pat), keys(md.metrics))

reset!(md::MetricLogger) = foreach(reset!, values(md.metrics))

function scores(md::MetricLogger)
    names = keys(md.metrics) .|> Symbol |> Tuple
    vals = map(compute, values(md.metrics))
    return NamedTuple{names}(vals)
end

"""
    Tracker(metrics...)

An object to track one or more metric values over the course of a training run.
Each metric can be either an `AbstractMetrics` or a `name => metric` pair.
In the first case, the default name of the provided metric will be used, while the
second allows us to choose an arbitrary name.

`Tracker` implements the `Tables.jl` interface, allowing it to be used as a table source.

# Example 1
```julia
julia> tracker = Tracker(Accuracy(), MIoU([0,1]));

julia> step!(tracker, [0, 0, 1, 0], [0, 1, 1, 0]);  # update all metrics

julia> scores(tracker)
(epoch = 1, accuracy = 0.75, MIoU = 0.5833333730697632)

julia> epoch!(tracker);

julia> tracker
Tracker(current_epoch=2)
┌───────┬──────────┬──────────┐
│ epoch │ accuracy │     MIoU │
│ Int64 │  Float64 │  Float64 │
├───────┼──────────┼──────────┤
│     1 │     0.75 │ 0.583333 │
│     2 │      0.0 │      0.0 │
└───────┴──────────┴──────────┘
```

# Example 2
```julia
julia> tracker = Tracker("train_acc" => Accuracy(), "val_acc" => Accuracy());

julia> step!(tracker, "train_acc", [0, 0, 1, 0], [0, 1, 1, 0]);  # specify metric to update

julia> step!(tracker, r"val_", [0, 0, 0, 0], [0, 1, 1, 0]);  # update metrics matching regex

julia> tracker
Tracker(current_epoch=1)
┌───────┬───────────┬─────────┐
│ epoch │ train_acc │ val_acc │
│ Int64 │   Float64 │ Float64 │
├───────┼───────────┼─────────┤
│     1 │      0.75 │     0.5 │
└───────┴───────────┴─────────┘
```
"""
struct Tracker{M}
    current::M
    history::Vector{M}
end

Tracker(metrics...; kw...) = Tracker(MetricLogger(metrics...; kw...))
Tracker(metrics::MetricLogger) = Tracker(metrics, Vector{typeof(metrics)}())

# Tables Interface
Tables.istable(::Type{Tracker}) = true
Tables.columnaccess(::Type{Tracker}) = true
Tables.rowaccess(::Type{Tracker}) = true
Tables.columns(x::Tracker) = Tables.rows(x) |> Tables.columntable
function Tables.rows(x::Tracker{M}) where {M}
    current = [(epoch=current_epoch(x), scores(x.current)...)]
    if current_epoch(x) > 1
        history = [(epoch=i, scores(metrics)...) for (i, metrics) in enumerate(x.history)]
        return vcat(history, current)
    end
    return current
end

"""
    step!(tracker::Tracker, ŷ, y)
    step!(tracker::Tracker, metric::String, ŷ, y)
    step!(tracker::Tracker, metric::Regex, ŷ, y)

Update the metric for the current epoch using the provided prediction/label pair.
"""
step!(tracker::Tracker, args...) = step!(tracker.current, args...)

"""
    epoch!(tracker::Tracker)

Store the metric value for the current epoch and reset the state.
"""
function epoch!(tracker::Tracker)
    push!(tracker.history, deepcopy(tracker.current))
    reset!(tracker.current)
end

"""
    best_epoch(tracker::Tracker, order::Order)

Returns the best epoch in `tracker` according to the given `order`.
"""
function best_epoch(t::Tracker, o::O) where {O <: Order}
    current_epoch(t) == 1 ? 1 : best_epoch(O, Tables.columns(t)[Symbol(o.metric)][1:end-1])
end
best_epoch(::Type{Max}, scores) = isempty(scores) ? 1 : argmax(scores)
best_epoch(::Type{Min}, scores) = isempty(scores) ? 1 : argmin(scores)

"""
    current_epoch(tracker::Tracker)

The current epoch in `tracker`.
"""
current_epoch(x::Tracker) = length(x.history) + 1

"""
    scores(t::Tracker; epoch=current_epoch(t), metrics=keys(t.metrics))

Return the metric scores for the provided epoch.
"""
function scores(t::Tracker; epoch=current_epoch(t), metrics=r".*")
    _current_epoch = current_epoch(t)
    if (epoch < 0) || (epoch > _current_epoch)
        throw(ArgumentError("Epoch $epoch is outside the range [0, $_current_epoch]!"))
    else
        _metrics = @match metrics begin
            x::Symbol => [x]
            x::String => [Symbol(x)]
            x::Regex => _filter_metrics(t.current, x) .|> Symbol
            x::Vector{String} => Symbol.(x)
            x::Vector{Symbol} => x
            _ => throw(ArgumentError("Invalid Metric Selector!"))
        end
        _scores = Tables.rowtable(t)[epoch]
        return _scores[(:epoch, _metrics...)]
    end
end

"""
    printscores(t::Tracker; epoch=current_epoch(t), metrics=r".*")

Return the metric scores for the provided epoch as a pretty-printed string.
"""
printscores(t::Tracker; kw...) = printscores(scores(t; kw...))
printscores(scores::NamedTuple) = join(["$m: $(_round(v))" for (m, v) in pairs(scores)], "  ")

_round(x::Int) = x
_round(x::AbstractFloat) = _round(Float64(x))
_round(x::Float64) = round(x, digits=4)

function Base.show(io::IO, ::MIME"text/plain", x::Tracker{M}) where {M}
    printstyled(io, "Tracker(current_epoch=$(current_epoch(x)))\n")
    PrettyTables.pretty_table(io, Tables.columntable(x))
end