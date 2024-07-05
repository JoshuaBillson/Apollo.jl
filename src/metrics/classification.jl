abstract type ClassificationMetric <: AbstractMetric end

function update(x::ClassificationMetric, state, ŷ::AbstractArray{<:AbstractFloat,N}, y::AbstractArray{<:AbstractFloat,N}) where {N}
    cdim = N - 1
    if size(ŷ, cdim) == 1
        return update(x, state, round.(Int, ŷ) .+ 1, round.(Int, y) .+ 1)
    end
    return update(x, state, mapslices(Flux.onecold, ŷ, dims=cdim), mapslices(Flux.onecold, y, dims=cdim))
end

# Accuracy

struct Accuracy <: ClassificationMetric end

function name(::Accuracy)
    return "accuracy"
end

function init(::Accuracy)
    return (correct=0, total=0)
end

function update(::Accuracy, state, ŷ::AbstractArray{Int}, y::AbstractArray{Int})
    return (correct = state.correct + sum(ŷ .== y), total = state.total + length(ŷ))
end

function compute(::Accuracy, state)
    state.correct / max(state.total, 1)
end

# MIoU

struct MIoU <: ClassificationMetric
    nclasses::Int
end

function name(::MIoU)
    return "MIoU"
end

function init(x::MIoU)
    return (intersection=zeros(Int, x.nclasses), union=zeros(Int, x.nclasses))
end

function update(x::MIoU, state, ŷ::AbstractArray{Int}, y::AbstractArray{Int})
    intersection = [sum((ŷ .== cls) .&& (y .== cls)) for cls in 1:x.nclasses]
    union = [sum((ŷ .== cls) .|| (y .== cls)) for cls in 1:x.nclasses]
    return (intersection = state.intersection .+ intersection, union = state.union .+ union)
end

function compute(x::MIoU, state)
    sum(state.intersection ./ (state.union .+ eps(Float32))) / x.nclasses
end