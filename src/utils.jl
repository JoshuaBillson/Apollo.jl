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

ones_like(x::AbstractArray{T}) where {T} = ones(T, size(x))

zeros_like(x::AbstractArray{T}) where {T} = zeros(T, size(x))

_crop(x::AbstractArray{<:Any,2}, xdims, ydims) = x[xdims,ydims]
_crop(x::AbstractArray{<:Any,3}, xdims, ydims) = x[xdims,ydims,:]
_crop(x::AbstractArray{<:Any,4}, xdims, ydims) = x[xdims,ydims,:,:]
_crop(x::AbstractArray{<:Any,5}, xdims, ydims) = x[xdims,ydims,:,:,:]
_crop(x::AbstractArray{<:Any,6}, xdims, ydims) = x[xdims,ydims,:,:,:,:]
_crop(x::HasDims, xdims, ydims) = x[X(xdims), Y(ydims)]

function _tile(x::AbstractArray, ul::Tuple{Int,Int}, tilesize::Tuple{Int,Int})
    # Compute Lower-Right Coordinates
    lr = ul .+ tilesize .- 1

    # Check Bounds
    any(tilesize .< 1) && throw(ArgumentError("Tile size must be positive!"))
    (any(ul .< 1) || any(lr .> _tilesize(x))) && throw(ArgumentError("Tile is out of bounds!"))

    # Crop Tile
    return _crop(x, ul[1]:lr[1], ul[2]:lr[2])
end

_tilesize(x::HasDims) = size.(Ref(x), (X,Y))
_tilesize(x::AbstractArray) = size(x)[1:2]

_permute(x, dims) = (Rasters.name(Rasters.dims(x)) == Rasters.name(dims)) ? x : permutedims(x, dims)

_stack(x::Vararg{Any}) = [x...]
_stack(x::Vararg{HasDims}) = [x...]
_stack(x::Vararg{AbstractArray{T,N}}) where {T,N} = cat(x..., dims=N)
_stack(x::AbstractVector{<:AbstractArray}) = _stack(x...)
_stack(x::AbstractVector{<:Tuple}) = _unzip(x) |> _stack
_stack(x::AbstractVector{<:Any}) = x
_stack(x::Tuple) = map(_stack, x)

_unzip(x::AbstractVector) = x
_unzip(x::AbstractVector{<:Tuple}) = map(f -> getfield.(x, f), fieldnames(eltype(x)))

_unsqueeze(x::AbstractArray) = reshape(x, size(x)..., 1)

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