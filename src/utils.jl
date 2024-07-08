apply(f, x) = f(x)
apply(f, x::AbstractVector) = map(f, x)

function catlayers(x::AbstractDimStack, ::Type{D}) where {D <: Rasters.Dimension}
    dim_vals = @pipe _dim_vals(x, D) |> _pretty_dim_vals(_, D)
    newdim = allunique(dim_vals) ? D(dim_vals) : D(1:length(dim_vals))
    return cat(layers(x)..., dims=newdim)
end

function foldlayers(f, xs::AbstractRasterStack)
    map(f ∘ skipmissing, layers(xs))
end

function folddims(f, xs::AbstractRaster; dims=Band)
    map(f ∘ skipmissing, eachslice(xs, dims=dims)).data
end

add_dim(raster, dims::Tuple) = reduce((acc, x) -> add_dim(acc, x), dims, init=raster)
function add_dim(x, ::Type{T}) where {T <: Rasters.DD.Dimension}
    if !hasdim(x, T)
        newdims = (dims(x)..., T(Base.OneTo(1))::T{Base.OneTo{Int64}})
        return Raster(reshape(x.data, (size(x)..., 1)), newdims)
    end
    return x
end

_unzip(x) = map(f -> getfield.(x, f), fieldnames(eltype(x)))

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

_dim_vals(x::AbstractDimStack, dim) = @pipe map(x -> _dim_vals(x, dim), layers(x)) |> reduce(vcat, _)
_dim_vals(x::AbstractDimArray, dim) = hasdim(x, dim) ? collect(dims(x, dim)) : Rasters.name(x)

function _pretty_dim_vals(vals::AbstractVector, dim)
    map(vals) do val
        @match val begin
            x::Int => Symbol("$(Rasters.name(dim))_$x")
            x::String => Symbol(x)
            x::Symbol => x
            _ => throw(ArgumentError("Non-Categorical Dimension!"))
        end
    end
end

_permute(x, dims) = (Rasters.name(Rasters.dims(x)) == Rasters.name(dims)) ? x : permutedims(x, dims)

_stack(x::Vararg{AbstractArray{T,N}}) where {T,N} = cat(x..., dims=N)

_missing_dims(x::AbstractDimArray, dims::Tuple) = filter(dim -> !hasdim(x, dim), dims)
_missing_dims(dims::Tuple, x::AbstractDimArray) = filter(dim -> !hasdim(dims, dim), Rasters.dims(x))

"Check that raster contains all of the dimensions in `dims` with no extra dimensions."
function _dims_match(raster::AbstractDimArray, dims)
    # Check That Raster Contains all Dims
    missing_dims = _missing_dims(raster, dims)
    if !isempty(missing_dims)
        throw(ArgumentError("Raster is missing dimension `$(Rasters.name(first(missing_dims)))`!"))
    end

    # Check That Raster Doesn't Have Extra Dims
    missing_dims = _missing_dims(dims, raster)
    if !isempty(missing_dims)
        throw(ArgumentError("Raster has extra dimension `$(Rasters.name(first(missing_dims)))`!"))
    end
end