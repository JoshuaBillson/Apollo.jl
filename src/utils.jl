tensor(rasters::AbstractVector; kwargs...) = tensor(rasters...; kwargs...)
function tensor(rasters::Vararg{AbstractRaster{<:Real}}; dims=nothing, precision=:f32)
    dims = isnothing(dims) ? Rasters.dims(first(rasters)) : dims
    tensors = map(x -> _tensor(x, dims), rasters)
    @assert _sizes_match(tensors...) "Tensor sizes do not match!"
    return @pipe _stack(tensors...) |> _precision(_, precision)
end

function raster(tensor::AbstractArray{<:Real,N}, tdims::Tuple, rdims...) where {N}
    @assert length(rdims) == size(tensor,N) "Mismatch between number of dims and observations!"
    rasters = [_raster(selectdim(tensor, N, i), tdims, rdim) for (i, rdim) in enumerate(rdims)]
    return length(rasters) == 1 ? first(rasters) : rasters
end

apply(f, x) = f(x)
apply(f, x::AbstractVector) = map(f, x)

function catlayers(x::AbstractRasterStack, dim)
    dim_names = _dim_names(x, dim)
    @assert allunique(dim_names) "Dimension values for new array are not unique!"
    return cat(layers(x)..., dims=dim(dim_names))
end

add_dim(raster, dims::Tuple) = reduce((acc, x) -> add_dim(acc, x), dims, init=raster)
function add_dim(x, ::Type{T}) where {T <: Rasters.DD.Dimension}
    if !hasdim(x, T)
        newdims = (dims(x)..., T(Base.OneTo(1))::T{Base.OneTo{Int64}})
        return Raster(reshape(x.data, (size(x)..., 1)), newdims)
    end
    return x
end

_precision(x::AbstractArray{Float16}, precision) = precision == :f16 ? x : _apply_precision(x, precision)
_precision(x::AbstractArray{Float32}, precision) = precision == :f32 ? x : _apply_precision(x, precision)
_precision(x::AbstractArray{Float64}, precision) = precision == :f64 ? x : _apply_precision(x, precision)
function _apply_precision(x::AbstractArray{<:Real}, precision)
    @match precision begin
        :f16 => Float16.(x)
        :f32 => Float32.(x)
        :f64 => Float64.(x)
    end
end

_dim_names(dim::Symbol) = dim
_dim_names(dim::String) = Symbol(dim)
_dim_names(dim::Int) = Symbol("Band_$dim")
_dim_names(dim::AbstractVector) = map(_dim_names, dim)
_dim_names(x::AbstractRasterStack, dim) = @pipe map(x -> _dim_names(x, dim), layers(x)) |> reduce(vcat, _)
function _dim_names(raster::AbstractRaster, dim)
    if hasdim(raster, dim) && (size(raster, dim) > 1)
        return _dim_names(collect(dims(raster, Band)))
    end
    return name(raster)
end

function _tensor(raster::AbstractRaster{<:Real,N}, dims) where {N}
    @assert allunique(dims) "dims must be unique!"
    @assert _has_dims(dims, raster) "dims is missing one or more dimensions of raster!"
    @assert _dims_match(raster, dims) "dims do not match dimensions of raster!"
    return @pipe _permute(raster, dims).data |> MLUtils.unsqueeze(_, dims=N+1)
end

function _raster(tensor::AbstractArray{<:Real,N}, tdims, rdims) where {N}
    dims = @match tdims begin
        (:X, :Y, :Band) || (X, Y, Band) => begin
            if size(tensor, 3) == length(Rasters.dims(rdims, Band))
                (Rasters.dims(rdims, X), Rasters.dims(rdims, Y), Rasters.dims(rdims, Band))
            else
                (Rasters.dims(rdims, X), Rasters.dims(rdims, Y), Band(1:size(tensor,3)))
            end
        end
    end
    return Raster(tensor, dims)
end

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