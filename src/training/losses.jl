"""
The super type of all loss functions. 

Subtypes need to implement a struct method with the call signature `(ŷ::AbstractArray, y::AbstractArray)`.
Instances can then be called as either `loss(ŷ, y)` or `loss(model, batch)`, where `batch` should
be a `Tuple` of the form `(x..., y)`.

# Example Implementation
```julia
struct MeanAbsoluteError <: AbstractLoss end

function (l::MeanAbsoluteError)(ŷ::AbstractArray, y::AbstractArray)
    return mean(abs.(ŷ .- y))
end
```
"""
abstract type AbstractLoss end

"""
The super type of all observation-weighted loss functions. 

Subtypes need to implement a struct method with the call signature 
`(ŷ::AbstractArray, y::AbstractArray, w::AbstractArray)`, where `w` is a mask that assigns an additional
weight term to each observation. Instances are called as either `loss(ŷ, y, w)` or 
`loss(model, batch)`, where `batch` should be a `Tuple` of the form `(x..., y, w)`.

# Example Implementation
```julia
struct WeightedMeanSquaredError <: WeightedLoss end

function (l::WeightedMeanSquaredError)(ŷ::AbstractArray, y::AbstractArray, weights::AbstractArray)
    return mean(((ŷ .- y) .^ 2) .* weights)
end
```
"""
abstract type WeightedLoss <: AbstractLoss end

(l::AbstractLoss)(m, batch::Tuple) = l(m(batch[1:end-1]...), batch[end])

(l::AbstractLoss)(ŷ::AbstractArray, y::AbstractArray) = compute_loss(l, ŷ, y)

(l::WeightedLoss)(m, batch::Tuple) = l(m(batch[1:end-2]...), batch[end-1], batch[end])

(l::WeightedLoss)(ŷ::AbstractArray, y::AbstractArray, w::AbstractArray) = compute_loss(l, ŷ, y, w)

struct CrossEntropy <: AbstractLoss end

function compute_loss(l::CrossEntropy, ŷ::AbstractArray{<:Real,4}, y::AbstractArray{<:Real,4})
    return compute_loss(l, _flatten(ŷ), _flatten(y))
end
function compute_loss(::CrossEntropy, ŷ::AbstractArray{T,2}, y::AbstractArray{T,2}) where {T <: Real}
    return mean(-sum(y .* log.(ŷ .+ _eps(T)); dims=1))
end

struct BinaryCrossEntropy <: AbstractLoss end

function compute_loss(::BinaryCrossEntropy, ŷ::AbstractArray{T}, y::AbstractArray{T}) where {T <: Real}
    return mean(-y .* log.(ŷ .+ _eps(T)) .- (1 .- y) .* log.(1 .- ŷ .+ _eps(T)))
end

struct WeightedBinaryCrossEntropy <: WeightedLoss end

function compute_loss(::WeightedBinaryCrossEntropy, ŷ::AbstractArray{T}, y::AbstractArray{T}, w::AbstractArray{T}) where {T <: Real}
    return mean(w .* (-y .* log.(ŷ .+ _eps(T)) .- (1 .- y) .* log.(1 .- ŷ .+ _eps(T))))
end

struct MeanAbsoluteError <: AbstractLoss end

compute_loss(::MeanAbsoluteError, ŷ::AbstractArray, y::AbstractArray) = mean(abs.(ŷ .- y))

struct MeanSquaredError <: AbstractLoss end

compute_loss(::MeanSquaredError, ŷ::AbstractArray, y::AbstractArray) = mean(((ŷ .- y) .^ 2))

struct BinaryDice <: AbstractLoss end

compute_loss(::BinaryDice, ŷ::AbstractArray, y::AbstractArray) = 1 - _dice_score(ŷ, y)

struct MultiClassDice <: AbstractLoss end

function compute_loss(l::MultiClassDice, ŷ::AbstractArray{<:Real,4}, y::AbstractArray{<:Real,4})
    return compute_loss(l, _flatten(ŷ), _flatten(y))
end
function compute_loss(::MultiClassDice, ŷ::AbstractArray{<:Real,2}, y::AbstractArray{<:Real,2})
    nclasses = size(y, 1)
    return mean([1 - _dice_score(ŷ[c,:], y[c,:]) for c in 2:nclasses])
end

struct MixedLoss{L1<:AbstractLoss,L2<:AbstractLoss,T<:AbstractFloat} <: AbstractLoss
    l1::L1
    l2::L2
    w::T
end

(l::MixedLoss)(ŷ::AbstractArray, y::AbstractArray) = (l.w * l.l1(ŷ, y)) + ((1 - l.w) * l.l2(ŷ, y))

"""
    MaskedLoss(loss::AbstractLoss)

Modifies a given loss function to only consider observed values, which are denoted by
a given mask, where `1` indicates that a value was observed and 0 indicates that it was not.

As with `WeightedLoss`, instances can be called as either `loss(ŷ, y, mask)` or `loss(model, batch)`, 
where `batch` should be a `Tuple` of the form `(x..., y, mask)`.
"""
struct MaskedLoss{L<:AbstractLoss} <: WeightedLoss
    loss::L
end

function (l::MaskedLoss)(ŷ::AbstractArray, y::AbstractArray, mask::AbstractArray)
    return l.loss(_select_obs(ŷ, mask), _select_obs(y, mask))
end

function _select_obs(x::AbstractArray{<:Real,4}, mask::AbstractArray{<:Real,4})
    x_flat = _flatten(x)
    indices = _flatten(mask) |> findall
    return x_flat[:,indices]
end

function _dice_score(ŷ::AbstractArray{T}, y::AbstractArray{T}) where {T <: Real}
    return (2 * sum(ŷ .* y) .+ _eps(T)) / (sum(ŷ .^ 2) + sum(y .^ 2) .+ _eps(T))
end

_eps(::Type{T}) where {T <: AbstractFloat} = eps(T)
_eps(::Type{T}) where {T <: Integer} = T(1)