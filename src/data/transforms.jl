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
    transform(t::AbstractTransform, dtype::DType, x)
    transform(t::AbstractTransform, dtypes::Tuple, x::Tuple)

Apply the transformation `t` to the input `x` with data type `dtype`.
"""
transform(t::AbstractTransform, dtype, data) = apply(t, dtype, data, rand(1:1000))
function transform(t::AbstractTransform, dtype, data::AbstractView)
    return MappedView(batch -> transform(t, dtype, batch), data)
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

Resample(scale::Number) = Resample(Float64(scale))

apply(t::Resample, ::Mask, x::HasDims, ::Int) = resample(x, t.scale, :near)
function apply(t::Resample, ::Image, x::HasDims, ::Int)
    return t.scale > 1 ? resample(x, t.scale, :bilinear) : resample(x, t.scale, :average)
end

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

(|>)(a::AbstractTransform, b::AbstractTransform) = ComposedTransform(a, b)
(|>)(a::ComposedTransform, b::AbstractTransform) = ComposedTransform(a.transforms..., b)
(|>)(a::AbstractTransform, b::ComposedTransform) = ComposedTransform(a, b.transforms...)
(|>)(a::ComposedTransform, b::ComposedTransform) = ComposedTransform(a.transforms..., b.transforms...)