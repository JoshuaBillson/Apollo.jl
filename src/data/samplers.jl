"""
    TileSampler(raster, tilesize::Int; stride=tilesize)

An object that iterates over tiles cut from a given raster.

# Parameters
- `raster`: An `AbstractRaster` or `AbstractRasterStack` to be cut into tiles.
- `tilesize`: The size (width and height) of the tiles.
- `stride`: The horizontal and vertical distance between adjacent tiles.
"""
struct TileSampler{D<:HasDims,TS}
    data::D
    tiles::Vector{Tuple{Int,Int}}
    tilesize::Int
end

function TileSampler(data::D, tilesize::Int; stride=tilesize) where {D<:HasDims}
    width, height = map(x -> size(data, x), (X,Y))
    xvals = 1:stride:width-tilesize+1
    yvals = 1:stride:height-tilesize+1
    tiles = [(x, y) for x in xvals for y in yvals]
    return TileSampler{D,tilesize}(data, tiles, tilesize)
end

Base.length(x::TileSampler) = length(x.tiles)

Base.getindex(x::TileSampler, i::Int) = getindex(x, [i]) |> first
Base.getindex(x::TileSampler, i::AbstractVector) = _tiles(x, i)

Base.iterate(x::TileSampler, state=1) = state > length(x) ? nothing : (x[state], state+1)

Base.firstindex(x::TileSampler) = 1
Base.lastindex(x::TileSampler) = length(x)

Base.keys(x::TileSampler) = Base.OneTo(length(x))

function Base.show(io::IO, x::TileSampler{D,TS}) where {D, TS}
    printstyled(io, "TileSampler(tile_size=$TS, num_tiles=$(length(x)))")
end

"""
    TileSeq(tiles)

An object that iterates over a `Vector` of tiles.
"""
struct TileSeq{D} <: TileSource
    tiles::D
end

function _tiles(s::TileSampler{D,TS}, i::AbstractVector) where {D,TS}
    return [s.data[X(x:x+TS-1), Y(y:y+TS-1)] for (x, y) in s.tiles[i]]
end