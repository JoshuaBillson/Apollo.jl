abstract type AbstractLoss end

struct BinaryCrossEntropy <: AbstractLoss end

function (l::BinaryCrossEntropy)(ŷ, y; weights=ones_like(y), mask=ones_like(y))
    l = Flux.binarycrossentropy(ŷ, y, agg=identity)
    total = sum(l .* weights .* mask)
    return total / sum(mask)
end

struct MeanAbsoluteError <: AbstractLoss end

function (l::MeanAbsoluteError)(ŷ, y; weights=ones_like(y), mask=ones_like(y))
    total = sum(abs.(ŷ .- y) .* weights .* mask)
    return total / sum(mask)
end

struct MeanSquaredError <: AbstractLoss end

function (l::MeanSquaredError)(ŷ, y; weights=ones_like(y), mask=ones_like(y))
    total = sum(((ŷ .- y) .^ 2) .* weights .* mask)
    return total / sum(mask)
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