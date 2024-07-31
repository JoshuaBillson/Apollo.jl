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
    return Raster(selectdim(tensor, N, 1), dims, missingval=T(missingval))
end

function resample(x::HasDims, scale, method=:bilinear)
    _check_resample_method(method)
    newsize = round.(Int, (size(x,X), size(x,Y)) .* scale)
    return Rasters.resample(x, size=newsize, method=method)
end

function upsample(x::AbstractArray, scale, method=:bilinear)
    @match method begin
        :linear => Flux.upsample_linear(x, scale)
        :bilinear => Flux.upsample_bilinear(x, scale)
        :trilinear => Flux.upsample_trilinear(x, scale)
        :nearest => Flux.upsample_nearest(x, scale)
        _ => throw(ArgumentError("`method` must be one of :linear, :bilinear, :trilinear, or :nearest!"))
    end
end

function resize(x::HasDims, newsize, method=:bilinear)
    _check_resample_method(method)
    return Rasters.resample(x, size=newsize, method=method)
end

"""
    crop(x::AbstractArray, size, ul=(1,1))

Crop a tile equal to `size` out of `x` with an upper-left corner defined by `ul`.
"""
crop(x::AbstractArray, size::Int, ul=(1,1)) = crop(x, (size, size), ul)
function crop(x::AbstractArray, size::Tuple{Int,Int}, ul=(1,1))
    return _crop(x, ul[1]:ul[1]+size[1]-1, ul[2]:ul[2]+size[2]-1)
end

function _check_resample_method(method)
    valid_methods = [:near, :bilinear, :cubic, :cubicspline, :lanczos, :average]
    if !(method in valid_methods)
        throw(ArgumentError("`method` must be one of $(join(map(x -> ":$x", valid_methods), ", ", ", or "))!"))
    end
end