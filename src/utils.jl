tensor(rasters::AbstractVector; kwargs...) = tensor(rasters...; kwargs...)
function tensor(rasters::Vararg{AbstractRaster{<:Real}}; dims=nothing)
    dims = isnothing(dims) ? Rasters.dims(first(rasters)) : dims
    tensors = map(x -> _tensor(x, dims), rasters)
    @assert _sizes_match(tensors...) "Tensor sizes do not match!"
    return _stack(tensors...)
end

function raster(tensor::AbstractArray{<:Real,N}, tdims::Tuple, rdims...) where {N}
    @assert length(rdims) == size(tensor,N) "Mismatch between number of dims and observations!"
    rasters = [_raster(selectdim(tensor, N, i), tdims, rdim) for (i, rdim) in enumerate(rdims)]
    return length(rasters) == 1 ? first(rasters) : rasters
end

apply(f, x) = f(x)
apply(f, x::AbstractVector) = map(f, x)

function catlayers(x::AbstractRasterStack, dim)
    if hasdim(x, dim)
        return catlayers(dropdims(x, dims=dim), dim)
    end
    raster = cat(layers(x)..., dims=dim)
    newdims = (dims(raster)[1:3]..., dim(collect(name(x))))
    Raster(raster, newdims)
end

function _tensor(raster::AbstractRaster{<:Real,N}, dims) where {N}
    @assert allunique(dims) "dims must be unique!"
    @assert _has_dims(dims, raster) "dims is missing one or more dimensions of raster!"
    @assert _dims_match(raster, dims) "dims do not match dimensions of raster!"
    return @pipe _permute(raster, dims).data |> MLUtils.unsqueeze(_, dims=N+1)
end

function _raster(tensor::AbstractArray{<:Real,N}, tdims, rdims) where {N}
    dims = @match tdims begin
        (:X, :Y, :Band) || (X, Y, Band) => (Rasters.dims(rdims, X), Rasters.dims(rdims, Y), Band(1:size(tensor,3)))
    end
    return Raster(tensor, dims)
end

function _add_dim(x, ::Type{T}) where T
    if !hasdim(x, T)
        newdims = (dims(x)..., T(Base.OneTo(1))::T{Base.OneTo{Int64}})
        return Raster(reshape(x.data, (size(x)..., 1)), newdims)
    end
    return x
end

_add_dims(raster, dims) = reduce((acc, x) -> _add_dim(acc, x), dims, init=raster)

_permute(x, dims) = (name(Rasters.dims(x)) == name(dims)) ? x : permutedims(x, dims)

_stack(x::Vararg{AbstractArray{T,N}}) where {T,N} = cat(x..., dims=N)

_missing_dims(x::AbstractRaster, dims) = tuple(filter(dim -> !hasdim(x, dim), dims)...)

_has_dims(raster::AbstractRaster, dims) = all(hasdim(raster, dims))
_has_dims(dims, raster::AbstractRaster) = all(hasdim(dims, Rasters.dims(raster)))

_dims_match(raster::AbstractRaster{<:Any,N}, dims) where {N} = (N == length(dims)) && _has_dims(raster, dims)

_sizes_match(xs...) = @pipe map(size, xs) |> all(==(first(_)), _)

function _name_to_dim(dim)
    @match dim begin
        :X => X
        :Y => Y
        :Z => Z
        :Band => Band
        x::Symbol => Dim{x}
    end
end