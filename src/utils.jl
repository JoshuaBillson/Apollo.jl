"""
    catlayers(x::AbstractRasterStack, dim)

Concatenate the layers of `x` along the dimension given by `dim`.
"""
function catlayers(x::AbstractRasterStack, ::Type{D}) where {D <: Rasters.Dimension}
    dim_vals = @pipe _dim_vals(x, D) |> _pretty_dim_vals(_, D)
    newdim = allunique(dim_vals) ? D(dim_vals) : D(1:length(dim_vals))
    return cat(layers(x)..., dims=newdim)
end

"""
    foldlayers(f, x::AbstractRasterStack)

Apply the reducer `f` to all non-missing elements in each layer of `x`.
"""
function foldlayers(f, x::AbstractRasterStack)
    map(f ∘ skipmissing, layers(x))
end

"""
    folddims(f, xs::AbstractRaster; dims=Band)

Apply the reducer function `f` to all non-missing elements in each slice of `x` WRT `dims`.

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
    putdim(raster::AbstractRaster, dims)

Add the provided singleton dim(s) to `raster`. Does nothing if `dims` is already present.

# Example
```julia
julia> r = Raster(rand(512,512), (X,Y));

julia> putdim(r, Band)
╭─────────────────────────────╮
│ 512×512×1 Raster{Float64,3} │
├─────────────────────────────┴─────────────────────────────── dims ┐
  ↓ X   ,
  → Y   ,
  ↗ Band Sampled{Int64} Base.OneTo(1) ForwardOrdered Regular Points
├─────────────────────────────────────────────────────────── raster ┤
  extent: Extent(X = (1, 512), Y = (1, 512), Band = (1, 1))

└───────────────────────────────────────────────────────────────────┘
[:, :, 1]
 0.107662  0.263251    0.786834  0.334663  …  0.316804   0.709557    0.478199
 0.379863  0.532268    0.635206  0.33514      0.402433   0.413602    0.657538
 0.129775  0.283808    0.327946  0.727027     0.685844   0.847777    0.435326
 0.73348   0.00705636  0.178885  0.381932     0.146575   0.310242    0.159852
 ⋮                                         ⋱                         ⋮
 0.330857  0.52704     0.888379  0.811084  …  0.0660687  0.00230472  0.448761
 0.698654  0.510846    0.916446  0.621061     0.23648    0.510697    0.113338
 0.600629  0.116626    0.567983  0.174267     0.089853   0.443758    0.667935
```
"""
putdim(raster::AbstractRaster, dims::Tuple) = reduce((acc, x) -> putdim(acc, x), dims, init=raster)
function putdim(raster::AbstractRaster, ::Type{T}) where {T <: Rasters.DD.Dimension}
    if !hasdim(raster, T)
        newdims = (dims(raster)..., T(Base.OneTo(1))::T{Base.OneTo{Int64}})
        return Raster(reshape(raster.data, (size(raster)..., 1)), newdims)
    end
    return raster
end

"""
    ones_like(x::AbstractArray)

Construct an array of ones with the same size and element type as `x`.
"""
ones_like(x::AbstractArray{T}) where {T} = ones(T, size(x))

"""
    zeros_like(x::AbstractArray)

Construct an array of zeros with the same size and element type as `x`.
"""
zeros_like(x::AbstractArray{T}) where {T} = zeros(T, size(x))

"""
    putobs(x::AbstractArray)

Add an N+1 obervation dimension of size 1 to the tensor `x`.
"""
putobs(x::AbstractArray) = reshape(x, size(x)..., 1)

"""
    dropobs(x::AbstractArray)

Remove the observation dimension from the tensor `x`.
"""
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

"""
    todevice(x)

Copy `x` to the GPU.
"""
todevice(x) = Flux.gpu(x)
todevice(x::AbstractRaster) = Rasters.modify(todevice, x)

"""
    stackobs(x...)
    stackobs(x::AbstractVector)

Stack the elements in `x` as if they were observations in a batch. If `x` is an `AbstractArray`, 
elements will be concatenated along the Nth dimension. Other data types will simply be placed
in a `Vector` in the same order as they were received. Special attention is paid to a `Vector` of
`Tuples`, where each tuple represents a single observation, such as a feature/label pair. In this
case, the tuples will first be unzipped, before their constituent elements are then stacked as usual.

# Example
```julia
julia> stackobs(1, 2, 3, 4, 5)
5-element Vector{Int64}:
 1
 2
 3
 4
 5

julia> stackobs([(1, :a), (2, :b), (3, :c)])
([1, 2, 3], [:a, :b, :c])

julia> stackobs([rand(256, 256, 3, 1) for _ in 1:10]...) |> size
(256, 256, 3, 10)

julia> xs = [rand(256, 256, 3, 1) for _ in 1:10];

julia> ys = [rand(256, 256, 1, 1) for _ in 1:10];

julia> data = collect(zip(xs, ys));

julia> stackobs(data) .|> size
((256, 256, 3, 10), (256, 256, 1, 10))
```
"""
stackobs(x::Vararg{Any}) = [x...]
stackobs(x::Vararg{HasDims}) = [x...]
stackobs(x::AbstractVector{<:Any}) = stackobs(x...)
stackobs(::Vararg{AbstractArray}) = throw(DimensionMismatch("Cannot stack tensors with different dimensions!"))
stackobs(x::Vararg{AbstractArray{T,N}}) where {T,N} = cat(x..., dims=N)
stackobs(x::AbstractVector{<:Tuple}) = @pipe unzip(x) |> map(stackobs, _)

"""
    unzip(x::AbstractVector{<:Tuple})

The reverse of `zip`.

# Example
```julia
julia> zip([1, 2, 3], [:a, :b, :c]) |> collect |> unzip
([1, 2, 3], [:a, :b, :c])
```
"""
unzip(x::AbstractVector) = x
unzip(x::AbstractVector{<:Tuple}) = map(f -> getfield.(x, f), fieldnames(eltype(x)))

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

_permute(x::AbstractRaster, dims) = (Rasters.name(Rasters.dims(x)) == Rasters.name(dims)) ? x : permutedims(x, dims)

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