"""
    tensor([precision], xs...; kwargs...)
    tensor([precision], x::AbstractArray; kwargs...)
    tensor([precision], x::AbstractRasterStack; layerdim=Band)

Convert one or more arrays to a tensor with an element type of `precision`.
`AbstractRasters` will be reshaped as necessary to enforce a dimension order of
(X,Y,Z,Band,Ti) before adding an observation dimension. Multiple arrays will be 
concatenated along the observation dimension after being converted to tensors.

# Parameters
- `precision`: Any `AbstractFloat` to use as the tensor's type (default = `Float32`).
- `x`: One or more `AbstractArrays` to be turned into tensors.
- `layerdim`: `AbstractRasterStacks` will have their layers concatenated along this dimension
before being turned into tensors.
"""
tensor(args...; kwargs...) = tensor(Float32, args...; kwargs...)
tensor(T::Type{<:AbstractFloat}, x::AbstractArray; kwargs...) = _precision(T, x)
tensor(T::Type{<:AbstractFloat}, xs::AbstractVector; kwargs...) = tensor(T, xs...; kwargs...)
tensor(T::Type{<:AbstractFloat}, xs...; kwargs...) = stackobs(map(x -> tensor(T, x; kwargs...), xs)...)
tensor(T::Type{<:AbstractFloat}, x::AbstractRasterStack; layerdim=Band) = tensor(T, catlayers(x, layerdim))
function tensor(T::Type{<:AbstractFloat}, x::AbstractRaster; kwargs...)
    if !hasdim(x, Band)  # Add Missing Band Dim
        return tensor(T, putdim(x, Band); precision=precision)
    end
    _dims = Rasters.commondims((X,Y,Z,Band,Ti), Rasters.dims(x))  # Enforce (X,Y,Z,Band,Ti) Order
    return @pipe _permute(x, _dims) |> _precision(T, putobs(_.data))
end

"""
    raster(tensor::AbstractArray, dims::Tuple; missingval=0)

Restore the raster dimensions given by `dims` to the provided tensor. The final dimension of
`tensor`, which is assumed to be the observation dimension, will be dropped.
"""
function raster(tensor::AbstractArray{T,N}, dims::Tuple; missingval=0) where {T,N}
    @assert length(dims) == (N - 1) "Tensor dims do not match raster dims"
    @assert size(tensor, N) == 1 "Cannot convert tensors with multiple observations!"
    sorted_dims = Rasters.dims(dims, Rasters.commondims((X,Y,Z,Band,Ti), dims))  # Enforce (X,Y,Z,Band,Ti) Order
    return Raster(selectdim(tensor, N, 1), sorted_dims, missingval=T(missingval))
end

"""
    normalize(x::AbstractArray, μ::AbstractVector, σ::AbstractVector; dim=3)

Normalize the input array with respect to the specified dimension so that the mean is 0
and the standard deviation is 1.

# Parameters
- `μ`: A `Vector` of means for each index in `dim`.
- `σ`: A `Vector` of standard deviations for each index in `dim`.
- `dim`: The dimension along which to normalize the input array.
"""
function normalize(x::AbstractArray{<:Real}, μ::AbstractArray{<:Real}, σ::AbstractArray{<:Real}; dim=3)
    return _normalize(x, μ, σ, dim)
end

_normalize(x::AbstractRaster, μ, σ, dim::Rasters.Dimension) = _normalize(x, μ, σ, Rasters.dimnum(x, dim))
_normalize(x::AbstractRaster, μ, σ, dim::Int) = Rasters.modify(x -> _normalize(x, μ, σ, dim), x)
function _normalize(x::AbstractArray, μ, σ, dim::Int)
    μ = vec2array(μ, x, dim)
    σ = vec2array(σ, x, dim)
    return (x .- μ) ./ σ
end

"""
    denormalize(x::AbstractArray, μ::AbstractVector, σ::AbstractVector; dim=3)

Denormalize the input array with respect to the specified dimension. Reverses the
effect of `normalize`.

# Parameters
- `μ`: A `Vector` of means for each index in `dim`.
- `σ`: A `Vector` of standard deviations for each index in `dim`.
- `dim`: The dimension along which to denormalize the input array.
"""
function denormalize(x::AbstractArray{<:Real}, μ::AbstractArray{<:Real}, σ::AbstractArray{<:Real}; dim=3)
    return _denormalize(x, μ, σ, dim)
end

_denormalize(x::AbstractRaster, μ, σ, dim::Rasters.Dimension) = _denormalize(x, μ, σ, Rasters.dimnum(x, dim))
_denormalize(x::AbstractRaster, μ, σ, dim::Int) = Rasters.modify(x -> _denormalize(x, μ, σ, dim), x)
function _denormalize(x::AbstractArray, μ, σ, dim::Int)
    μ = vec2array(μ, x, dim)
    σ = vec2array(σ, x, dim)
    return (x .* σ) .+ μ
end

"""
    resample(x::AbstractRaster, scale::AbstractFloat, method=:bilinear)
    resample(x::AbstractRasterStack, scale::AbstractFloat, method=:bilinear)

Resample `x` according to the given `scale` and `method`.

# Parameters
- `x`: The raster/stack to be resampled.
- `scale`: The size of the output with respect to the input.
- `method`: One of `:nearest`, `:bilinear`, `:cubic`, `:cubicspline`, `:lanczos`, or `:average`.
"""
resample(x::AbstractArray, args...) = Apollo.resample(_array_to_raster(x), args...).data
function resample(x::HasDims, scale, method=:bilinear)
    (scale <= 0) && throw(ArgumentError("scale must be strictly greater than zero (received $scale)."))
    _check_resample_method(method)
    newsize = round.(Int, (size(x,X), size(x,Y)) .* scale)
    return Rasters.resample(x, size=newsize, method=method == :nearest ? :near : method)
end

_array_to_raster(x::AbstractArray{<:Any,4}) = Rasters.Raster(x, (X,Y,Band,Dim{:obs}))
_array_to_raster(x::AbstractArray{<:Any,5}) = Rasters.Raster(x, (X,Y,Band,Ti,Dim{:obs}))
_array_to_raster(x::AbstractArray{<:Any,6}) = Rasters.Raster(x, (X,Y,Z,Band,Ti,Dim{:obs}))

"""
    upsample(x::AbstractArray, scale, method=:bilinear)

Upsample the array `x` according to the given `scale` and `method`. This function
is specialized for tensors, and will generally be much faster than `resample`.

# Parameters
- `x`: The array to be upsampled.
- `scale`: A positive `Integer` specifying the size of the output with respect to the input.
- `method`: One of `:bilinear` or `:nearest`.
"""
upsample(::HasDims, args...) = throw(ArgumentError("upsample can only be called on tensors!"))
upsample(x::AbstractArray{<:Real}, scale, args...) = upsample(x, _scale(x, scale), args...)
function upsample(x::AbstractArray{<:Real,N}, scale::NTuple{S,<:Real}, method=:bilinear) where {N,S}
    @assert all(>=(1), scale) && all(x -> x isa Integer, scale) "Scale must be a positive non-zero integer!"
    if (N - 2) != S 
        throw(ArgumentError("The scale argument should be an NTuple with length $(N-2), but it has length $S."))
    else
        @match method begin
            :bilinear => Flux.upsample_bilinear(x, scale)
            :nearest => Flux.upsample_nearest(x, scale)
            _ => throw(ArgumentError("`method` must be one of :bilinear or :nearest!"))
        end
    end
end

_scale(::AbstractArray{<:Any,4}, scale::T) where {T <: Real} = (scale, scale)
_scale(::AbstractArray{<:Any,5}, scale::T) where {T <: Real} = (scale, scale, T(1))
_scale(::AbstractArray{<:Any,6}, scale::T) where {T <: Real} = (scale, scale, T(1), T(1))

"""
    resize(x::AbstractRaster, newsize::Tuple{Int,Int}, method=:bilinear)
    resize(x::AbstractRasterStack, newsize::Tuple{Int,Int}, method=:bilinear)

Resize the raster/stack `x` to `newsize` under the specified `method`.

# Parameters
- `x`: The array to be resized.
- `newsize`: The width and height of the output as a tuple.
- `method`: One of `:nearest`, `:bilinear`, `:cubic`, `:cubicspline`, `:lanczos`, or `:average`.
"""
function resize(x::HasDims, newsize::Tuple{Int,Int}, method=:bilinear)
    _check_resample_method(method)
    return Rasters.resample(x, size=newsize, method=method)
end

"""
    crop(x, size::Int, ul=(1,1))
    crop(x, size::Tuple{Int,Int}, ul=(1,1))

Crop a tile equal to `size` out of `x` with an upper-left corner defined by `ul`.
"""
crop(x, size::Int, ul=(1,1)) = crop(x, (size, size), ul)
crop(x::HasDims, size::Tuple{Int,Int}, ul=(1,1)) = _tile(x, ul, size)
crop(x::AbstractArray, size::Tuple{Int,Int}, ul=(1,1)) = _tile(x, ul, size)

_crop(x::AbstractArray{<:Any,2}, xdims, ydims) = x[xdims,ydims]
_crop(x::AbstractArray{<:Any,3}, xdims, ydims) = x[xdims,ydims,:]
_crop(x::AbstractArray{<:Any,4}, xdims, ydims) = x[xdims,ydims,:,:]
_crop(x::AbstractArray{<:Any,5}, xdims, ydims) = x[xdims,ydims,:,:,:]
_crop(x::AbstractArray{<:Any,6}, xdims, ydims) = x[xdims,ydims,:,:,:,:]
_crop(x::HasDims, xdims, ydims) = x[X(xdims), Y(ydims)]

function _tile(x, ul::Tuple{Int,Int}, tilesize::Tuple{Int,Int})
    # Compute Lower-Right Coordinates
    lr = ul .+ tilesize .- 1

    # Check Bounds
    any(tilesize .< 1) && throw(ArgumentError("Tile size must be positive!"))
    (any(ul .< 1) || any(lr .> _tilesize(x))) && throw(ArgumentError("Tile is out of bounds!"))

    # Crop Tile
    return _crop(x, ul[1]:lr[1], ul[2]:lr[2])
end

function _check_resample_method(method)
    valid_methods = [:nearest, :bilinear, :cubic, :cubicspline, :lanczos, :average]
    if !(method in valid_methods)
        throw(ArgumentError("`method` must be one of $(join(map(x -> ":$x", valid_methods), ", ", ", or "))!"))
    end
end