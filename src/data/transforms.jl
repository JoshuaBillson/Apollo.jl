import Base.|>
import Base.*

abstract type AbstractTransform end

abstract type DType{N} end
struct Image{N} <: DType{N} end
struct Mask{N} <: DType{N} end

Image() = Image(:x)
Image(x::Symbol) = Image{x}()

Mask() = Mask(:y)
Mask(x::Symbol) = Mask{x}()

"""
    transform(t::AbstractTransform, dtype::DType, x)
    transform(t::AbstractTransform, dtypes::Tuple, x::Tuple)

Apply the transformation `t` to the input `x` with data type `dtype`.
"""
transform(::AbstractTransform, ::DType, data) = data
function transform(t::AbstractTransform, dtypes::Tuple, data::Tuple)
    @assert length(dtypes) == length(data) "Number of types don't match number of sources!"
    if all(Apollo.is_tile_source, data)
        return MappedView(batch -> transform(t, dtypes, batch), data)
    end
    return Tuple(transform(t, dtypes[i], data[i]) for i in eachindex(data))
end

# Tensor Transform

struct Tensor{D,T} <: AbstractTransform
    layerdim::Type{D}
    precision::Type{T}
end

function Tensor(;precision=Float32, layerdim=Band)
    return Tensor(layerdim, precision)
end

function transform(t::Tensor, ::DType, x)
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

Flux.@layer Normalize trainable=()

(n::Normalize)(x::AbstractArray) = normalize(x, n.μ, n.σ)

transform(t::Normalize{Nothing}, ::Image, x) = normalize(x, t.μ, t.σ)
transform(t::Normalize, ::Image, x) = normalize(x, t.μ, t.σ, dim=t.dim)

# DeNormalize Transform

struct DeNormalize{D} <: AbstractTransform
    dim::D
    μ::Vector{Float64}
    σ::Vector{Float64}
end

function DeNormalize(μ, σ; dim=nothing)
    return DeNormalize(dim, Float64.(μ), Float64.(σ))
end

Flux.@layer DeNormalize trainable=()

(n::DeNormalize)(x::AbstractArray) = denormalize(x, n.μ, n.σ)

transform(t::DeNormalize{Nothing}, ::Image, x) = denormalize(x, t.μ, t.σ)
transform(t::DeNormalize{Int}, ::Image, x) = denormalize(x, t.μ, t.σ, dim=t.dim)

# Resample Transform

struct Resample <: AbstractTransform
    scale::Int
end

transform(t::Resample, ::Mask, x::HasDims) = resample(x, t.scale, :near)
function transform(t::Resample, ::Image, x::HasDims)
    return t.scale > 1 ? resample(x, t.scale, :bilinear) : resample(x, t.scale, :average)
end

# Filtered Transform

struct FilteredTransform{D,T} <: AbstractTransform
    dtype::D
    transform::T
end

function transform(t::FilteredTransform{D1,T}, dtype::D2, x) where {D1<:DType,D2<:DType,T}
    D1 == D2 ? transform(t.transform, dtype, x) : x
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

function transform(t::ComposedTransform, dtype::DType, x::AbstractArray)
    return reduce((acc, trans) -> transform(trans, dtype, acc), t.transforms, init=x)
end

(|>)(a::AbstractTransform, b::AbstractTransform) = ComposedTransform(a, b)
(|>)(a::ComposedTransform, b::AbstractTransform) = ComposedTransform(a.transforms..., b)
(|>)(a::AbstractTransform, b::ComposedTransform) = ComposedTransform(a, b.transforms...)
(|>)(a::ComposedTransform, b::ComposedTransform) = ComposedTransform(a.transforms..., b.transforms...)