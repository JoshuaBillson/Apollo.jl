abstract type RegressionMetric <: AbstractMetric end

# Loss Tracking

struct Loss{L} <: RegressionMetric
    loss::L
end

name(::Loss) = "loss"

init(::Loss) = (total=0.0, n=0)

function update(x::Loss, state, ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    return (total = state.total + x.loss(ŷ, y), n = state.n + 1)
end

compute(::Loss, state) = state.total / max(state.n, 1)

# Mean Squared Error

struct MSE <: RegressionMetric end

name(::MSE) = "mse"

init(::MSE) = (total=0.0, n=0)

function update(::MSE, state, ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    return (total = state.total + (ŷ .- y) .^ 2, n = state.n + length(y))
end

compute(::MSE, state) = state.total / max(state.n, 1)