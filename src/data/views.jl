# Interface

"""
Super type of all iterators.
"""
abstract type AbstractView{D} end

data(x::AbstractView) = x.data

is_tile_source(::Any) = false
is_tile_source(x::AbstractView) = is_tile_source(data(x))
is_tile_source(x::AbstractView{<:Tuple}) = all(map(is_tile_source, data(x)))

Base.getindex(x::AbstractView, i::AbstractVector) = map(j -> getindex(x, j), i) |> stackobs

Base.iterate(x::AbstractView, state=1) = state > length(x) ? nothing : (x[state], state+1)

Base.firstindex(x::AbstractView) = 1
Base.lastindex(x::AbstractView) = length(x)

Base.keys(x::AbstractView) = Base.OneTo(length(x))

# MappedView

"""
    MappedView(f, data)

An iterator which lazily applies `f` to each element in `data` when requested.
"""
struct MappedView{F<:Function,D} <: AbstractView{D}
    f::F
    data::D
end

MappedView(f::Function, data::Tuple) = MappedView(f, ObsView(data, eachindex(first(data))))

Base.length(x::MappedView) = length(data(x))

Base.getindex(x::MappedView, i::Int) = x.f(x.data[i])

# JoinedView

"""
    JoinedView(data...)

An object that iterates over each element in the iterators given by `data` as
if they were concatenated into a single list.
"""
struct JoinedView{D} <: AbstractView{D}
    data::D
    JoinedView(data) = JoinedView((data,))
    JoinedView(data...) = JoinedView(data)
    JoinedView(data::D) where {D <: Tuple} = new{D}(data)
end

Base.length(x::JoinedView) = map(length, x.data) |> sum

function Base.getindex(x::JoinedView, i::Int)
    (i > length(x) || i < 1) &&  throw(BoundsError(x, i)) 
    lengths = map(length, x.data) |> cumsum
    for (j, len) in enumerate(lengths)
        if i <= len
            offset = j == 1 ? 0 : lengths[j-1]
            return  x.data[j][i-offset]
        end
    end
end

# ObsView

"""
    ObsView(data, indices)

Construct an iterator over the elements specified by `indices` in `data`.
"""
struct ObsView{D} <: AbstractView{D}
    data::D
    indices::Vector{Int}

    ObsView(data, indices::AbstractVector{Int}) = ObsView(data, collect(indices))
    function ObsView(data::D, indices::Vector{Int}) where {D}
        _check_indices(data, indices)
        new{D}(data, indices)
    end
end

function _check_indices(data, indices)
    for index in indices
        !(index in eachindex(data)) && throw(ArgumentError("Index $index not found!"))
    end
end

Base.length(x::ObsView) = length(x.indices)

Base.getindex(x::ObsView, i::Int) = data(x)[x.indices[i]]

"""
    ZippedView(data...)

Construct an iterator that zips each element of the given subiterators into a `Tuple`.
"""
struct ZippedView{D} <: AbstractView{D}
    data::D

    ZippedView(data...) = ZippedView(tuple(data...))
    function ZippedView(data::D) where {D<:Tuple}
        @assert _all_equal(eachindex, data) "Iterators have different indices!"
        return new{D}(data)
    end
end

Base.length(x::ZippedView) = data(x) |> first |> length

Base.getindex(x::ZippedView, i::Int) = map(d -> d[i], data(x))

"""
    TileView(raster, tilesize::Int; stride=tilesize)

An object that iterates over tiles cut from a given raster.

# Parameters
- `raster`: An `AbstractRaster` or `AbstractRasterStack` to be cut into tiles.
- `tilesize`: The size (width and height) of the tiles.
- `stride`: The horizontal and vertical distance between adjacent tiles.
"""
struct TileView{D<:HasDims,TS} <: AbstractView{D}
    data::D
    tiles::Vector{Tuple{Int,Int}}
    tilesize::Int
end

function TileView(data::D, tilesize::Int; stride=tilesize) where {D<:HasDims}
    width, height = map(x -> size(data, x), (X,Y))
    xvals = 1:stride:width-tilesize+1
    yvals = 1:stride:height-tilesize+1
    tiles = [(x, y) for x in xvals for y in yvals]
    return TileView{D,tilesize}(data, tiles, tilesize)
end

Base.length(x::TileView) = length(x.tiles)

function Base.getindex(t::TileView{D,TS}, i::Int) where {D,TS}
    (x, y) = t.tiles[i]
    return t.data[X(x:x+TS-1), Y(y:y+TS-1)]
end

# Methods

"""
    splitobs(data; at=0.8, shuffle=true)

Return a set of indices that splits the given observations according to the given break points.

# Arguments
- `data`: Any type that implements `Base.length()`. 
- `at`: The fractions at which to split `data`. 
- `shuffle`: If `true`, shuffles the indices before splitting. 

# Example
```julia
julia> splitobs(1:100, at=(0.7, 0.2), shuffle=false)
3-element Vector{Vector{Int64}}:
 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
 [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
 [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
```
"""
splitobs(data; kwargs...) = map(x -> ObsView(data, x), splitobs(eachindex(data); kwargs...))
function splitobs(data::AbstractVector{Int}; at=0.8, shuffle=true)
    sum(at) >= 1 && throw(ArgumentError("'at' cannot sum to more than 1!"))
    indices = shuffle ? Random.randperm(length(data)) : collect(1:length(data))
    breakpoints = _breakpoints(length(data), at)
    starts = (1, (breakpoints .+ 1)...)
    ends = ((breakpoints)..., length(data))
    return [indices[s:e] for (s, e) in zip(starts, ends)]
end

_breakpoints(n::Int, at::Tuple) = round.(Int, cumsum(at) .* n)
_breakpoints(n::Int, at::Real) = _breakpoints(n, (at,))
_breakpoints(n::Int, at::AbstractVector) = _breakpoints(n, Tuple(at))

"""
    zipobs(data...)

Create a new iterator where the elements of each iterator in `data` are returned as a tuple.

# Example
```jldoctest
julia> zipobs(1:5, 41:45, [:a, :b, :c, :d, :e]) |> collect
5-element Vector{Any}:
 (1, 41, :a)
 (2, 42, :b)
 (3, 43, :c)
 (4, 44, :d)
 (5, 45, :e)
```
"""
zipobs(data...) = zipobs(tuple(data...))
zipobs(data::Tuple) = ZippedView(data)

"""
    repeatobs(data, n::Int)

Create a new view which iterates over every element in `data` `n` times.
"""
repeatobs(data, n::Int) = JoinedView([data for _ in 1:n]...)

"""
    takeobs(data, obs::AbstractVector{Int})

Take all observations from `data` whose index corresponds to `obs` while removing everything else.
"""
takeobs(data, obs::AbstractVector{Int}) = ObsView(data, obs)

"""
    dropobs(data, obs::AbstractVector{Int})

Remove all observations from `data` whose index corresponds to those in `obs`.
"""
dropobs(data, obs::AbstractVector{Int}) = takeobs(data, filter(x -> !(x in obs), eachindex(data)))

"""
    filterobs(f, data)

Remove all observations from `data` for which `f` returns `false`.
"""
filterobs(f, data) = takeobs(data, findall(map(f, data)))

"""
    mapobs(f, data)

Lazily apply the function `f` to each element in `data`.
"""
mapobs(f, data) = MappedView(f, data)

"""
    sampleobs([rng], data, n)

Randomly sample `n` elements from `data` without replacement. `rng` may be optionally
provided for reproducible results.
"""
sampleobs(rng, data, n::Int) = takeobs(data, Random.shuffle(rng, eachindex(data))[1:n])
sampleobs(data, n::Int) = takeobs(data, Random.shuffle(eachindex(data))[1:n])

"""
    shuffleobs([rng], data)

Randomly shuffle the elements of `data`. Provide `rng` for reproducible results.
"""
shuffleobs(rng, data) = takeobs(data, Random.shuffle(rng, eachindex(data)))
shuffleobs(data) = takeobs(data, Random.shuffle(eachindex(data)))