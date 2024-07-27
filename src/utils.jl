function catlayers(x::AbstractDimStack, ::Type{D}) where {D <: Rasters.Dimension}
    dim_vals = @pipe _dim_vals(x, D) |> _pretty_dim_vals(_, D)
    newdim = allunique(dim_vals) ? D(dim_vals) : D(1:length(dim_vals))
    return cat(layers(x)..., dims=newdim)
end

function foldlayers(f, xs::AbstractRasterStack)
    map(f ∘ skipmissing, layers(xs))
end

"""
    folddims(f, xs::AbstractRaster; dim=Band)

Reduce the collection of non-missing values to a singular value in each slice of `x` WRT `dims`.

# Arguments
- `f`: A function that reduces an array of values to a singular value.
- `x`: An `AbstractRaster` over which we want to fold.
- `dims`: The dimension used to generate each slice that is passed to `f`.

# Example
```julia
julia> μ = folddims(mean, raster, dims=Band)
6-element Vector{Float32}:
 0.09044644
 0.23737456
 0.30892986
 0.33931717
 0.16186203
 0.076255515
```
"""
function folddims(f, x::AbstractRaster; dims=Band)
    map(f ∘ skipmissing, eachslice(x, dims=dims)).data
end

"""
    putdim(raster::AbstractRaster, dims::Tuple)
    putdim(x::AbstractRaster, ::Type{Rasters.DD.Dimension})

Add the provided singleton dim(s) to the given raster. Does nothing if `dim` is already present.
"""
putdim(raster::AbstractRaster, dims::Tuple) = reduce((acc, x) -> putdim(acc, x), dims, init=raster)
function putdim(raster::AbstractRaster, ::Type{T}) where {T <: Rasters.DD.Dimension}
    if !hasdim(raster, T)
        newdims = (dims(raster)..., T(Base.OneTo(1))::T{Base.OneTo{Int64}})
        return Raster(reshape(raster.data, (size(raster)..., 1)), newdims)
    end
    return raster
end

ones_like(x::AbstractArray{T}) where {T} = ones(T, size(x))

zeros_like(x::AbstractArray{T}) where {T} = zeros(T, size(x))

putobs(x::AbstractArray) = reshape(x, size(x)..., 1)

function dropobs(x::AbstractArray{<:Any,N}) where {N}
    @assert size(x,N) == 1 "Cannot drop dimension with multiple observations!"
    dropdims(x, dims=N)
end

"""
    vec2array(x::AbstractVector, to::AbstractArray, dim::Int)

Reshape the vector `x` to have the same number of dimensions as `to`. Missing dimensions 
are added as singletons while the dimension corresponding to `dim` will be filled with the
values of `x`.
"""
function vec2array(x::AbstractVector, to::AbstractArray{T,N}, dim::Int) where {T,N}
    @assert 0 < dim <= N
    @assert size(to, dim) == length(x)
    newshape = Tuple([i == dim ? length(x) : 1 for i in 1:N])
    return reshape(x, newshape)
end

todevice(x::AbstractRaster) = Rasters.modify(Flux.gpu, x)
todevice(x::AbstractArray) = Flux.gpu(x)

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
_stack(x::Vararg{AbstractArray}) = throw(DimensionMismatch("Cannot stack tensors with different dimensions!"))
_stack(x::Vararg{AbstractArray{T,N}}) where {T,N} = cat(x..., dims=N)
_stack(x::AbstractVector{<:Any}) = _stack(x...)
_stack(x::AbstractVector{<:Tuple}) = @pipe _unzip(x) |> map(_stack, _)

_unzip(x::AbstractVector) = x
_unzip(x::AbstractVector{<:Tuple}) = map(f -> getfield.(x, f), fieldnames(eltype(x)))

_precision(::Type{T}, x::AbstractArray{T}) where {T} = x 
_precision(::Type{T}, x::AbstractArray) where {T} = T.(x)

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