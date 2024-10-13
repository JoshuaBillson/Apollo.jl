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
    hasdim(dims, Band) || return raster(tensor, (dims..., Band); missingval=missingval)
    @assert length(dims) == (N - 1) "Tensor dims do not match raster dims"
    @assert size(tensor, N) == 1 "Cannot convert tensors with multiple observations!"
    sorted_dims = Rasters.dims(dims, Rasters.commondims((X,Y,Z,Band,Ti), dims))  # Enforce (X,Y,Z,Band,Ti) Order
    return Raster(selectdim(tensor, N, 1), sorted_dims, missingval=T(missingval))
end

"""
    onehot([precision], logits::AbstractArray, labels)

Encode `logits` as a one-hot encoded tensor with the specified `precision`.

# Parameters
- `precision`: Any `AbstractFloat` to use as the tensor's type (default = `Float32`).
- `labels`: The labels to use in the one-hot encoding. The first label will be assigned to the first index,
the second label to the second index, and so on.
"""
onehot(logits::AbstractArray, labels) = onehot(Float32, logits, labels)
onehot(T::Type{<:AbstractFloat}, logits::AbstractRaster, labels) = onehot(T, tensor(logits), labels)
onehot(T::Type{<:AbstractFloat}, logits::AbstractVector, labels) = onehot(T, reshape(logits, (1,:)), labels)
onehot(T::Type{<:AbstractFloat}, logits::AbstractArray{<:Real,2}, labels) = T.(_onehot(logits, labels, 1))
onehot(T::Type{<:AbstractFloat}, logits::AbstractArray{<:Real,4}, labels) = T.(_onehot(logits, labels, 3))

function _onehot(logits::AbstractArray, labels, dim::Int)
    @assert size(logits, dim) == 1
    return cat(map(label -> logits .== label, labels)..., dims=dim)
end

onecold(x::AbstractArray{T,2}) where {T <: Real} = T.(_onecold(x, 1))
onecold(x::AbstractArray{T,4}) where {T <: Real} = T.(_onecold(x, 3))

_onecold(x::AbstractArray, dim::Int) = mapslices(argmax, x, dims=dim) .- 1

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
crop(x::HasDims, size::Tuple{Int,Int}, ul=(1,1)) = _crop(x, ul, size)
crop(x::AbstractArray, size::Tuple{Int,Int}, ul=(1,1)) = _crop(x, ul, size)

_crop(x::AbstractArray{<:Any,2}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims]
_crop(x::AbstractArray{<:Any,3}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:]
_crop(x::AbstractArray{<:Any,4}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:,:]
_crop(x::AbstractArray{<:Any,5}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:,:,:]
_crop(x::AbstractArray{<:Any,6}, xdims::AbstractVector, ydims::AbstractVector) = x[xdims,ydims,:,:,:,:]
_crop(x::HasDims, xdims::AbstractVector, ydims::AbstractVector) = x[X(xdims), Y(ydims)]
function _crop(x, ul::Tuple{Int,Int}, tilesize::Tuple{Int,Int})
    # Compute Lower-Right Coordinates
    lr = ul .+ tilesize .- 1

    # Check Bounds
    any(tilesize .< 1) && throw(ArgumentError("Tile size must be positive!"))
    (any(ul .< 1) || any(lr .> _tilesize(x))) && throw(ArgumentError("Tile is out of bounds!"))

    # Crop Tile
    return _crop(x, ul[1]:lr[1], ul[2]:lr[2])
end

"""
    flipX(x)

Flip the image `x` across the horizontal axis.
"""
flipX(x::HasDims) = x[Y(size(x,Y):-1:1)]
flipX(x::AbstractArray{T,2}) where {T} = x[:,end:-1:1]
flipX(x::AbstractArray{T,3}) where {T} = x[:,end:-1:1,:]
flipX(x::AbstractArray{T,4}) where {T} = x[:,end:-1:1,:,:]
flipX(x::AbstractArray{T,5}) where {T} = x[:,end:-1:1,:,:,:]
flipX(x::AbstractArray{T,6}) where {T} = x[:,end:-1:1,:,:,:,:]

"""
    flipY(x)

Flip the image `x` across the vertical axis.
"""
flipY(x::HasDims) = x[X(size(x,X):-1:1)]
flipY(x::AbstractArray{T,2}) where {T} = x[end:-1:1,:]
flipY(x::AbstractArray{T,3}) where {T} = x[end:-1:1,:,:]
flipY(x::AbstractArray{T,4}) where {T} = x[end:-1:1,:,:,:]
flipY(x::AbstractArray{T,5}) where {T} = x[end:-1:1,:,:,:,:]
flipY(x::AbstractArray{T,6}) where {T} = x[end:-1:1,:,:,:,:,:]

"""
    rot90(x)

Rotate the image `x` by 90 degress. 

**Note:** `x` must be a square image.
"""
function rot90(x::AbstractArray{<:Real,4})
    return @pipe permutedims(x, (2, 1, 3, 4)) |> reverse(_, dims=2)
end

function _check_resample_method(method)
    valid_methods = [:nearest, :bilinear, :cubic, :cubicspline, :lanczos, :average]
    if !(method in valid_methods)
        throw(ArgumentError("`method` must be one of $(join(map(x -> ":$x", valid_methods), ", ", ", or "))!"))
    end
end