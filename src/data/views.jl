"""
    TileView(raster, tilesize::Int; stride=tilesize)

An object that iterates over tiles cut from a given raster.

# Parameters
- `raster`: An `AbstractRaster` or `AbstractRasterStack` to be cut into tiles.
- `tilesize`: The size (width and height) of the tiles.
- `stride`: The horizontal and vertical distance between adjacent tiles.
"""
struct TileView{D<:HasDims,TS}
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

Base.getindex(t::TileView, i::AbstractVector) = [getindex(t, j) for j in i]
function Base.getindex(t::TileView{D,TS}, i::Int) where {D,TS}
    (x, y) = t.tiles[i]
    return t.data[X(x:x+TS-1), Y(y:y+TS-1)]
end

function Base.show(io::IO, x::TileView{D,TS}) where {D, TS}
    printstyled(io, "TileSampler(tile_size=$TS, num_tiles=$(length(x)))")
end