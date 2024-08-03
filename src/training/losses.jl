abstract type AbstractLoss end

struct BinaryCrossEntropy <: AbstractLoss end

function (l::BinaryCrossEntropy)(ŷ, y)
    return Flux.binarycrossentropy(ŷ, y)
end

function (l::BinaryCrossEntropy)(ŷ, y, weights)
    l = Flux.binarycrossentropy(ŷ, y, agg=identity)
    return mean(l .* weights)
end

struct MeanAbsoluteError <: AbstractLoss end

function (l::MeanAbsoluteError)(ŷ, y)
    return mean(abs.(ŷ .- y))
end

function (l::MeanAbsoluteError)(ŷ, y, weights)
    return mean(abs.(ŷ .- y) .* weights)
end

struct MeanSquaredError <: AbstractLoss end

function (l::MeanSquaredError)(ŷ, y)
    return mean(((ŷ .- y) .^ 2))
end

function (l::MeanSquaredError)(ŷ, y, weights)
    return mean(((ŷ .- y) .^ 2) .* weights)
end

struct DiceLoss <: AbstractLoss end

function (l::DiceLoss)(ŷ, y)
    Flux.dice_coeff_loss(ŷ, y)
end

struct MixedLoss{L1<:AbstractLoss,L2<:AbstractLoss,T<:AbstractFloat} <: AbstractLoss
    l1::L1
    l2::L2
    w::T
end

(l::MixedLoss)(ŷ, y) = (l.w * l.l1(ŷ, y)) + ((1 - l.w) * l.l2(ŷ, y))

struct MaskedLoss{L<:AbstractLoss}
    loss::L
end

function (l::MaskedLoss)(ŷ::AbstractArray{T}, y::AbstractArray{T}, mask) where {T}
    indices = findall(==(1), mask)
    return isempty(indices) ? T(0.0) : l.loss(ŷ[indices], y[indices])
end