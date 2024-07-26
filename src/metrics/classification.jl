abstract type ClassificationMetric <: AbstractMetric end

function update(x::ClassificationMetric, state, ŷ::AbstractArray{<:AbstractFloat,N}, y::AbstractArray{<:AbstractFloat,N}) where {N}
    cdim = N - 1
    if size(ŷ, cdim) == 1
        return update(x, state, round.(Int, ŷ), round.(Int, y))
    end
    return update(x, state, mapslices(Flux.onecold, ŷ, dims=cdim), mapslices(Flux.onecold, y, dims=cdim))
end

# Accuracy

struct Accuracy <: ClassificationMetric end

name(::Type{Accuracy}) = "accuracy"

init(::Accuracy) = (correct=0, total=0)

function update(::Accuracy, state, ŷ::AbstractArray{Int}, y::AbstractArray{Int})
    return (correct = state.correct + sum(ŷ .== y), total = state.total + length(ŷ))
end

compute(::Accuracy, state) = state.correct / max(state.total, 1)

# MIoU

struct MIoU <: ClassificationMetric
    nclasses::Int
end

name(::Type{MIoU}) = "MIoU"

init(x::MIoU) = (intersection=zeros(Int, x.nclasses), union=zeros(Int, x.nclasses))

function update(x::MIoU, state, ŷ::AbstractArray{Int}, y::AbstractArray{Int})
    intersection = [sum((ŷ .== cls) .&& (y .== cls)) for cls in 1:x.nclasses]
    union = [sum((ŷ .== cls) .|| (y .== cls)) for cls in 1:x.nclasses]
    return (intersection = state.intersection .+ intersection, union = state.union .+ union)
end

compute(x::MIoU, state) = sum(state.intersection ./ (state.union .+ eps(Float32))) / x.nclasses