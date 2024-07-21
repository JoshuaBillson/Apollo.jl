import Base.|>
import Base.*

abstract type AbstractTransform end
abstract type RandomTransform <: AbstractTransform end

Image() = Image(:x)
Image(x::Symbol) = Image{x}()

Mask() = Mask(:y)
Mask(x::Symbol) = Mask{x}()

"""
    transform(t::AbstractTransform, dtype::DType, x)
    transform(t::AbstractTransform, dtypes::Tuple, x::Tuple)

Apply the transformation `t` to the input `x` with data type `dtype`.
"""

transform(t::AbstractTransform, dtype, data) = apply(t, dtype, data, rand(1:1000))
function transform(t::AbstractTransform, dtype, data::AbstractIterator)
    return MappedView(batch -> transform(t, dtype, batch), data)
end

apply(::AbstractTransform, ::DType, data, ::Int) = data
function apply(t::AbstractTransform, dtypes::Tuple, data::Tuple, seed::Int)
    @assert length(dtypes) == length(data) "Number of types don't match number of sources!"
    return Tuple(apply(t, dtypes[i], data[i], seed) for i in eachindex(data))
end

# Tensor Transform

struct Tensor{D,T} <: AbstractTransform
    layerdim::Type{D}
    precision::Type{T}
end

function Tensor(;precision=Float32, layerdim=Band)
    return Tensor(layerdim, precision)
end

function apply(t::Tensor, ::DType, x, ::Int)
    return tensor(t.precision, x; layerdim=t.layerdim)
end

# Normalize Transform

struct Normalize{D} <: AbstractTransform
    dim::D
    μ::Vector{Float64}
    σ::Vector{Float64}
end

function Normalize(μ, σ; dim=nothing)
    return Normalize(dim, Float64.(μ), Float64.(σ))
end

apply(t::Normalize{Nothing}, ::Image, x, ::Int) = normalize(x, t.μ, t.σ)
apply(t::Normalize, ::Image, x, ::Int) = normalize(x, t.μ, t.σ, dim=t.dim)

# DeNormalize Transform

struct DeNormalize{D} <: AbstractTransform
    dim::D
    μ::Vector{Float64}
    σ::Vector{Float64}
end

function DeNormalize(μ, σ; dim=nothing)
    return DeNormalize(dim, Float64.(μ), Float64.(σ))
end

apply(t::DeNormalize{Nothing}, ::Image, x, ::Int) = denormalize(x, t.μ, t.σ)
apply(t::DeNormalize{Int}, ::Image, x, ::Int) = denormalize(x, t.μ, t.σ, dim=t.dim)

# Resample Transform

struct Resample <: AbstractTransform
    scale::Int
end

apply(t::Resample, ::Mask, x::HasDims, ::Int) = resample(x, t.scale, :near)
function apply(t::Resample, ::Image, x::HasDims, ::Int)
    return t.scale > 1 ? resample(x, t.scale, :bilinear) : resample(x, t.scale, :average)
end

# Crop Transform

struct Crop <: AbstractTransform
    size::Tuple{Int,Int}
end

Crop(size::Int) = Crop((size, size))

function apply(t::Crop, ::DType, x::AbstractArray, ::Int)
    return crop(x, t.size)
end

# Random Crop

struct RandomCrop <: RandomTransform
    size::Tuple{Int,Int}
end

function apply(t::RandomCrop, ::DType, x::AbstractArray, seed::Int)
    xpad = size(x, 1) - t.size[1]
    ypad = size(x, 2) - t.size[2]
    outcome = Tuple(rand(Random.MersenneTwister(seed), Random.uniform(Float64), 2))
    ul = max.(ceil.(Int, (xpad, ypad) .* outcome), 1)
    return crop(x, t.size, ul)
end

# Filtered Transform

struct FilteredTransform{D,T} <: AbstractTransform
    dtype::D
    transform::T
end

function apply(t::FilteredTransform{D1,T}, dtype::D2, x, seed::Int) where {D1<:DType,D2<:DType,T}
    D1 == D2 ? apply(t.transform, dtype, x, seed) : x
end

(*)(dtype::DType, t::AbstractTransform) = FilteredTransform(dtype, t)
(*)(dtypes::Tuple, t::AbstractTransform) = ComposedTransform(map(dtype -> dtype * t, dtypes)...)

# Composed Transform

struct ComposedTransform{T} <: AbstractTransform
    transforms::T

    function ComposedTransform(transforms::Vararg{AbstractTransform})
        return new{typeof(transforms)}(transforms)
    end
end

function apply(t::ComposedTransform, dtype::DType, x, seed::Int)
    return reduce((acc, trans) -> apply(trans, dtype, acc, seed), t.transforms, init=x)
end

(|>)(a::AbstractTransform, b::AbstractTransform) = ComposedTransform(a, b)
(|>)(a::ComposedTransform, b::AbstractTransform) = ComposedTransform(a.transforms..., b)
(|>)(a::AbstractTransform, b::ComposedTransform) = ComposedTransform(a, b.transforms...)
(|>)(a::ComposedTransform, b::ComposedTransform) = ComposedTransform(a.transforms..., b.transforms...)