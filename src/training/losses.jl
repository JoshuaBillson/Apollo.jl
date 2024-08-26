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

function (l::AbstractLoss)(m, batch::Tuple)
    return l(m(batch[1:end-1]...), batch[end])
end

function (l::WeightedLoss)(m, batch::Tuple)
    return l(m(batch[1:end-2]...), batch[end-1], batch[end])
end

struct BinaryCrossEntropy <: AbstractLoss end

function (l::BinaryCrossEntropy)(ŷ::AbstractArray, y::AbstractArray)
    return Flux.binarycrossentropy(ŷ, y)
end

struct WeightedBinaryCrossEntropy <: WeightedLoss end

function (l::WeightedBinaryCrossEntropy)(ŷ::AbstractArray, y::AbstractArray, weights::AbstractArray)
    l = Flux.binarycrossentropy(ŷ, y, agg=identity)
    return mean(l .* weights)
end

struct MeanAbsoluteError <: AbstractLoss end

function (l::MeanAbsoluteError)(ŷ::AbstractArray, y::AbstractArray)
    return mean(abs.(ŷ .- y))
end

struct WeightedMeanAbsoluteError <: WeightedLoss end

function (l::WeightedMeanAbsoluteError)(ŷ::AbstractArray, y::AbstractArray, weights::AbstractArray)
    return mean(abs.(ŷ .- y) .* weights)
end

struct MeanSquaredError <: AbstractLoss end

function (l::MeanSquaredError)(ŷ::AbstractArray, y::AbstractArray)
    return mean(((ŷ .- y) .^ 2))
end

struct WeightedMeanSquaredError <: WeightedLoss end

function (l::WeightedMeanSquaredError)(ŷ::AbstractArray, y::AbstractArray, weights::AbstractArray)
    return mean(((ŷ .- y) .^ 2) .* weights)
end

struct BinaryDice <: AbstractLoss end

function (l::BinaryDice)(ŷ::AbstractArray, y::AbstractArray)
    Flux.dice_coeff_loss(ŷ, y)
end

struct MultiClassDice <: AbstractLoss end

function (l::MultiClassDice)(ŷ::AbstractArray{<:Real,4}, y::AbstractArray{<:Real,4})
    nclasses = size(y, 3)
    ŷ_flat = @pipe permutedims(ŷ, (3, 1, 2, 4)) |> reshape(_, (nclasses, :))
    y_flat = @pipe permutedims(y, (3, 1, 2, 4)) |> reshape(_, (nclasses, :))
    l(ŷ_flat, y_flat)
end
function (l::MultiClassDice)(ŷ::AbstractArray{<:Real,2}, y::AbstractArray{<:Real,2})
    nclasses = size(y, 1)
    return mean([Flux.dice_coeff_loss(ŷ[c,:], y[c,:]) for c in 2:nclasses])
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
    flattened = @pipe permutedims(x, (3,1,2,4)) |> reshape(_, (size(x,3), :))
    indices = @pipe permutedims(mask, (3,1,2,4)) |> reshape(_, (:)) |> findall
    return flattened[:,indices]
end