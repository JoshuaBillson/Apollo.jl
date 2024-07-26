abstract type Order end

struct Max <: Order 
    metric::String
end

struct Min <: Order 
    metric::String
end

struct MetricDict{D<:OrderedDict}
    metrics::D
end

function MetricDict(metrics::Vararg{Pair{String,<:AbstractMetric}})
    metric_dict = OrderedDict{String,Any}()
    for (name, metric) in metrics
        metric_dict[name] = Metric(metric)
    end
    return MetricDict(metric_dict)
end

Base.keys(x::MetricDict) = keys(x.metrics)
Base.getindex(x::MetricDict, i) = x.metrics[i]
Base.setindex!(x::MetricDict, val, key...) = Base.setindex!(x.metrics, val, keys...)
Base.pairs(x::MetricDict) = Base.pairs(x.metrics)

struct Tracker{M}
    metrics::M
    history::OrderedDict{String,Vector}
end

Tracker(metrics::Vararg{Pair{String,<:AbstractMetric}}) = Tracker(MetricDict(metrics...))
function Tracker(metrics::MetricDict)
    history = OrderedDict{String,Vector}("epoch"=>Int[])
    foreach(name -> setindex!(history, Float64[], name), keys(metrics))
    return Tracker(metrics, history)
end

# Tables Interface
Tables.istable(::Type{Tracker}) = true
Tables.columnaccess(::Type{Tracker}) = true
Tables.rowaccess(::Type{Tracker}) = true
Tables.rows(x::Tracker) = Tables.columns(x) |> Tables.rowtable
function Tables.columns(x::Tracker) 
    history = deepcopy(x.history)
    for k in keys(history)
        if k == "epoch"
            history[k] = vcat(history["epoch"], current_epoch(x))
        else
            history[k] = vcat(history[k], compute(x.metrics[k]))
        end
    end
    return history |> Tables.columntable
end

metrics(tracker::Tracker) = tracker.metrics

"""
    step!(tracker::Tracker, ŷ, y)

Update the metric's state for the current batch of predictions and labels.
"""
step!(tracker::Tracker, metric::String, ŷ, y) = update!(tracker.metrics[metric], ŷ, y)
function step!(tracker::Tracker, pattern::Regex, ŷ, y)
    foreach(metric -> step!(tracker, metric, ŷ, y), _filter_metrics(tracker, pattern))
end

"""
    epoch!(tracker::Tracker)

Store the metric value for the current epoch and reset the state.
"""
function epoch!(tracker::Tracker)
    epochs = tracker.history["epoch"]
    push!(epochs, maximum(last, epochs, init=0.0) + 1.0)
    for (name, metric) in pairs(tracker.metrics)
        push!(tracker.history[name], compute(metric))
        reset!(metric)
    end
end

best_epoch(t::Tracker, o::O) where {O <: Order} = best_epoch(O, t.history[o.metric])
best_epoch(::Type{Max}, scores) = isempty(scores) ? 1 : argmax(scores)
best_epoch(::Type{Min}, scores) = isempty(scores) ? 1 : argmin(scores)

current_epoch(x::Tracker) = maximum(length, values(x.history), init=0) + 1

function scores(t::Tracker; epoch=current_epoch(t), metrics=keys(t.metrics))
    _current_epoch = current_epoch(t)
    if (epoch < 0) || (epoch > _current_epoch)
        throw(ArgumentError("Epoch $epoch is outside the range [0, $_current_epoch]!"))
    else
        _metrics = @match metrics begin
            x::Symbol => [x]
            x::String => [Symbol(x)]
            x::Regex => _filter_metrics(t, metrics) .|> Symbol
            x => Symbol.(x)
        end
        _scores = Tables.rowtable(t)[epoch]
        return _scores[(:epoch, _metrics...)]
    end
end

printscores(t::Tracker; epoch=current_epoch(t), metrics=keys(t.metrics)) = printscores(scores(t; epoch=epoch, metrics=metrics))
printscores(scores::NamedTuple) = join(["$m: $(_round(v))" for (m, v) in pairs(scores)], "  ")

_round(x::Int) = x
_round(x::AbstractFloat) = _round(Float64(x))
_round(x::Float64) = round(x, digits=4)

function Base.show(io::IO, ::MIME"text/plain", x::Tracker{M}) where {M}
    printstyled(io, "Tracker(current_epoch=$(current_epoch(x)))\n")
    PrettyTables.pretty_table(io, Tables.columntable(x))
end

_filter_metrics(t::Tracker, pat::Regex) = filter(x -> contains(x, pat), keys(t.metrics))