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