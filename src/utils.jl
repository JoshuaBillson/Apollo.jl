
apply(f, x) = f(x)
apply(f, x::AbstractVector) = map(f, x)

function catlayers(x::AbstractRasterStack, dim)
    dim_names = _dim_names(x, dim)
    dim_names = allunique(dim_names) ? dim_names : 1:length(dim_names)
    #@assert allunique(dim_names) "Dimension values for new array are not unique!"
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
_precision(x::AbstractArray{<:Real}, precision) = _apply_precision(x, precision)
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

_permute(x, dims) = (name(Rasters.dims(x)) == name(dims)) ? x : permutedims(x, dims)

_stack(x::Vararg{AbstractArray{T,N}}) where {T,N} = cat(x..., dims=N)

_missing_dims(x::AbstractDimArray, dims::Tuple) = filter(dim -> !hasdim(x, dim), dims)
_missing_dims(dims::Tuple, x::AbstractDimArray) = filter(dim -> !hasdim(dims, dim), Rasters.dims(x))

_has_dims(raster::AbstractRaster, dims) = all(hasdim(raster, dims))
_has_dims(dims, raster::AbstractRaster) = all(hasdim(dims, Rasters.dims(raster)))

function _dims_match(raster::AbstractDimArray, dims)
    # Check That Raster Contains all Dims
    missing_dims = _missing_dims(raster, dims)
    if !isempty(missing_dims)
        throw(ArgumentError("Raster is missing dimension `$(name(first(missing_dims)))`!"))
    end

    # Check That Raster Doesn't Have Extra Dims
    missing_dims = _missing_dims(dims, raster)
    if !isempty(missing_dims)
        throw(ArgumentError("Raster has extra dimension `$(name(first(missing_dims)))`!"))
    end
end

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