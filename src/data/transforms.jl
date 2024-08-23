import Base.|>
import Base.*

"""
Super type of all transforms.
"""
abstract type AbstractTransform end

"""
Super type of all random transforms.
"""
abstract type RandomTransform <: AbstractTransform end

"""
    TransformedView(data, dtype, transform::AbstractTransform).

An iterator that applied the provided `transform` to each batch in data.
The transform will modify each element according to the specified `dtype`.
"""
struct TransformedView{D,DT,T} <: AbstractIterator{D}
    data::D
    dtype::DT
    transform::T
end

Base.length(x::TransformedView) = length(data(x))

Base.getindex(x::TransformedView, i::Int) = apply(x.transform, x.dtype, x.data[i], rand(1:1000))

"""
    transform(t::AbstractTransform, dtype::DType, x)
    transform(t::AbstractTransform, dtypes::Tuple, x::Tuple)

Apply the transformation `t` to the input `x` with data type `dtype`.
"""
transform(t::AbstractTransform, dtype, data) = apply(t, dtype, data, rand(1:1000))
function transform(t::AbstractTransform, dtype, data::AbstractIterator)
    return TransformedView(data, dtype, t)
end

"""
    apply(t::AbstractTransform, dtype::DType, data, seed)

Apply the transformation `t` to the `data` of type `dtype` with the random seed `seed`.
"""
apply(::AbstractTransform, ::DType, data, ::Int) = data
function apply(t::AbstractTransform, dtypes::Tuple, data::Tuple, seed::Int)
    @assert length(dtypes) == length(data) "Number of types don't match number of sources!"
    return Tuple(apply(t, dtypes[i], data[i], seed) for i in eachindex(data))
end

# Tensor Transform

"""
    Tensor(;precision=Float32, layerdim=Band)

Convert raster/stack into a tensor with the specified `precision`. See `tensor` for more details.

# Parameters
- `precision`: Any `AbstractFloat` to use as the tensor's type (default = `Float32`).
- `layerdim`: `RasterStacks` will have their layers concatenated along this dimension.
"""
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

description(x::Tensor) = "Transform to tensor."

# Normalize Transform

"""
    Normalize(μ, σ; dim=3)

Normalize the input array with respect to the specified dimension so that the mean is 0
and the standard deviation is 1.

# Parameters
- `μ`: A `Vector` of means for each index in `dim`.
- `σ`: A `Vector` of standard deviations for each index in `dim`.
- `dim`: The dimension along which to normalize the input array.
"""
struct Normalize <: AbstractTransform
    dim::Int
    μ::Vector{Float64}
    σ::Vector{Float64}
end

function Normalize(μ, σ; dim=3)
    return Normalize(dim, Float64.(μ), Float64.(σ))
end

apply(t::Normalize, ::Image, x, ::Int) = normalize(x, t.μ, t.σ, dim=t.dim)

description(x::Normalize) = "Normalize by channel dimension."

# DeNormalize Transform

"""
    DeNormalize(μ, σ; dim=3)

Denormalize the input array with respect to the specified dimension. Reverses the
effect of `normalize`.

# Parameters
- `μ`: A `Vector` of means for each index in `dim`.
- `σ`: A `Vector` of standard deviations for each index in `dim`.
- `dim`: The dimension along which to denormalize the input array.
"""
struct DeNormalize <: AbstractTransform
    dim::Int
    μ::Vector{Float64}
    σ::Vector{Float64}
end

function DeNormalize(μ, σ; dim=3)
    return DeNormalize(dim, Float64.(μ), Float64.(σ))
end

apply(t::DeNormalize, ::Image, x, ::Int) = denormalize(x, t.μ, t.σ, dim=t.dim)

description(x::DeNormalize) = "Denormalize by channel dimension."

# Resample Transform

"""
    Resample(scale)

Resample `x` according to the specified `scale`. `Mask` types will always be
resampled with `:near` interpolation, whereas `Images` will be resampled with 
either `:bilinear` (`scale` > `1`) or `:average` (`scale` < `1`).

# Parameters
- `x`: The raster/stack to be resampled.
- `scale`: The size of the output with respect to the input.
"""
struct Resample <: AbstractTransform
    scale::Float64
end

function Resample(scale::Number)
    (scale <= 0) && throw(ArgumentError("scale must be strictly greater than zero (received $scale)."))
    Resample(Float64(scale))
end

apply(t::Resample, ::Mask, x, ::Int) = resample(x, t.scale, :nearest)
function apply(t::Resample, ::Image, x, ::Int)
    return t.scale > 1 ? resample(x, t.scale, :bilinear) : resample(x, t.scale, :average)
end

description(x::Resample) = "Resample by a factor of $(x.scale)."

# Crop Transform

"""
    Crop(size::Int)
    Crop(size::Tuple{Int,Int})

Crop a tile equal to `size` out of the input array with an upper-left corner at (1,1).
"""
struct Crop <: AbstractTransform
    size::Tuple{Int,Int}
end

Crop(size::Int) = Crop((size, size))

function apply(t::Crop, ::DType, x::AbstractArray, ::Int)
    return crop(x, t.size)
end

description(x::Crop) = "Crop tiles to $(first(x.size))x$(last(x.size))."

# Random Crop

"""
    RandomCrop(size::Int)
    RandomCrop(size::Tuple{Int,Int})

Crop a randomly placed tile equal to `size` from the input array.
"""
struct RandomCrop <: RandomTransform
    size::Tuple{Int,Int}
end

RandomCrop(size::Int) = RandomCrop((size, size))

function apply(t::RandomCrop, ::DType, x::AbstractArray, seed::Int)
    xpad = size(x, 1) - t.size[1]
    ypad = size(x, 2) - t.size[2]
    outcome = Tuple(rand(Random.MersenneTwister(seed), Random.uniform(Float64), 2))
    ul = max.(ceil.(Int, (xpad, ypad) .* outcome), 1)
    return crop(x, t.size, ul)
end

description(x::RandomCrop) = "Random crop to $(first(x.size))x$(last(x.size))."

# FlipX

"""
    FlipX(p)

Apply a random horizontal flip with probability `p`.
"""
struct FlipX <: AbstractTransform
    p::Float64

    FlipX(p::Real) = FlipX(Float64(p))
    function FlipX(p::Float64)
        (0 <= p <= 1) || throw(ArgumentError("p must be between 0 and 1!"))
        return new(p)
    end
end

apply(t::FlipX, ::DType, x, seed::Int) = _apply_random(seed, t.p) ? flipX(x) : x

description(x::FlipX) = "Random horizontal flip with probability $(round(x.p, digits=2))."

# FlipY

"""
    FlipY(p)

Apply a random vertical flip with probability `p`.
"""
struct FlipY <: AbstractTransform
    p::Float64

    FlipY(p::Real) = FlipY(Float64(p))
    function FlipY(p::Float64)
        (0 <= p <= 1) || throw(ArgumentError("p must be between 0 and 1!"))
        return new(p)
    end
end

apply(t::FlipY, ::DType, x, seed::Int) = _apply_random(seed, t.p) ? flipY(x) : x

description(x::FlipY) = "Random vertical flip with probability $(round(x.p, digits=2))."

# Rot90

"""
    Rot90(p)

Apply a random 90 degree rotation with probability `p`.
"""
struct Rot90 <: AbstractTransform
    p::Float64

    Rot90(p::Real) = Rot90(Float64(p))
    function Rot90(p::Float64)
        (0 <= p <= 1) || throw(ArgumentError("p must be between 0 and 1!"))
        return new(p)
    end
end

apply(t::Rot90, ::DType, x, seed::Int) = _apply_random(seed, t.p) ? rot90(x) : x

description(x::Rot90) = "Random 90 degree rotation with probability $(round(x.p, digits=2))."

# Filtered Transform

"""
    FilteredTransform(dtype::DType, t::AbstractTransform)

Modify the transform `t` so that it will only be applied to inputs whose
type and name matches `dtype`. The `*` operator is overloaded for convenience.

# Example
```julia
julia> r = Raster(rand(256,256, 3), (X,Y,Band));

julia> t = Image(:x2) * Resample(2.0);

julia> apply(t, Image(), r, 123) |> size
(256, 256, 3)

julia> apply(t, Image(:x2), r, 123) |> size
(512, 512, 3)

julia> apply(t, Mask(:x2), r, 123) |> size
(256, 256, 3)

```
"""
struct FilteredTransform{D<:DType,T<:AbstractTransform} <: AbstractTransform
    dtype::D
    transform::T
end

function apply(t::FilteredTransform{D1,T}, dtype::D2, x, seed::Int) where {D1<:DType,D2<:DType,T}
    D1 == D2 ? apply(t.transform, dtype, x, seed) : x
end

description(x::FilteredTransform) = "$(description(x.transform)) ($(x.dtype))"

(*)(dtype::DType, t::AbstractTransform) = FilteredTransform(dtype, t)
(*)(dtypes::Tuple, t::AbstractTransform) = ComposedTransform(map(dtype -> dtype * t, dtypes)...)

# Composed Transform

"""
    ComposedTransform(transforms...)

Apply `transforms` to the input in the same order as they are given.

# Example
```julia
julia> r = Raster(rand(256,256, 3), (X,Y,Band));

julia> t = Resample(2.0) |> Tensor();

julia> apply(t, Image(), r, 123) |> size
(512, 512, 3, 1)

julia> apply(t, Image(), r, 123) |> typeof
Array{Float32, 4}
```
"""
struct ComposedTransform{T} <: AbstractTransform
    transforms::T

    function ComposedTransform(transforms::Vararg{AbstractTransform})
        return new{typeof(transforms)}(transforms)
    end
end

function apply(t::ComposedTransform, dtype::DType, x, seed::Int)
    return reduce((acc, trans) -> apply(trans, dtype, acc, seed), t.transforms, init=x)
end

function Base.show(io::IO, x::ComposedTransform)
    print(io, "$(length(x.transforms))-step ComposedTransform:")
    for (i, t) in enumerate(x.transforms) 
     print(io, "\n  $i) $(description(t))")
    end
end

(|>)(a::AbstractTransform, b::AbstractTransform) = ComposedTransform(a, b)
(|>)(a::ComposedTransform, b::AbstractTransform) = ComposedTransform(a.transforms..., b)
(|>)(a::AbstractTransform, b::ComposedTransform) = ComposedTransform(a, b.transforms...)
(|>)(a::ComposedTransform, b::ComposedTransform) = ComposedTransform(a.transforms..., b.transforms...)

function _apply_random(seed::Int, p::Float64)
    @assert 0 <= p <= 1 "p must be between 0 and 1!"
    outcome = rand(Random.MersenneTwister(seed), Random.uniform(Float64))
    return outcome <= p
end