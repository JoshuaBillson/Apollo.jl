"""
Classification metrics are used to evaluate the performance of models that predict a
discrete label for each observation. Subtypes should implement an `update` method which
assumes that both `ŷ` and `y` are encoded as logits.
"""
abstract type ClassificationMetric <: AbstractMetric end

function update(x::ClassificationMetric, state, ŷ::AbstractArray{<:AbstractFloat,N}, y::AbstractArray{<:Real,N}) where {N}
    if N == 1
        return update(x, state, reshape(ŷ, (1,:)), reshape(y, (1,:)))
    else
        cdim = N - 1
        if size(ŷ, cdim) == 1
            return update(x, state, round.(Int, ŷ), round.(Int, y))
        end
        return update(x, state, mapslices(Flux.onecold, ŷ, dims=cdim), mapslices(Flux.onecold, y, dims=cdim))
    end
end

# Accuracy

"""
    Accuracy()
    
Measures the model's overall accuracy as `correct / total`.
"""
struct Accuracy <: ClassificationMetric end

name(::Type{Accuracy}) = "accuracy"

init(::Accuracy) = (correct=0, total=0)

function update(::Accuracy, state, ŷ::AbstractArray{Int}, y::AbstractArray{Int})
    (correct = state.correct + sum(ŷ .== y), total = state.total + length(ŷ))
end

compute(::Accuracy, state) = state.correct / max(state.total, 1)

# MIoU

"""
    MIoU(classes::Vector{Int})

Mean Intersection over Union (MIoU) is a measure of the overlap between a prediction and a label.
This measure is frequently used for segmentation models.
"""
struct MIoU <: ClassificationMetric
    classes::Vector{Int}
end

MIoU(x::AbstractVector) = MIoU(vec(x))

name(::Type{MIoU}) = "MIoU"

init(x::MIoU) = (intersection=zeros(Int, length(x.classes)), union=zeros(Int, length(x.classes)))

function update(x::MIoU, state, ŷ::AbstractArray{Int}, y::AbstractArray{Int})
    intersection = [sum((ŷ .== cls) .&& (y .== cls)) for cls in x.classes]
    union = [sum((ŷ .== cls) .|| (y .== cls)) for cls in x.classes]
    return (intersection = state.intersection .+ intersection, union = state.union .+ union)
end

compute(x::MIoU, state) = sum((state.intersection .+ eps(Float64)) ./ (state.union .+ eps(Float64))) / length(x.classes)