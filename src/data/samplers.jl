struct TileSampler{D,TS}
    data::D
    tiles::Vector{Tuple{Int,Int}}
    tilesize::Int

    function TileSampler(data::D, tilesize::Int; kwargs...) where {D}
        return TileSampler{D,tilesize}(data, tilesize; kwargs...)
    end
    function TileSampler{D,TS}(data, tilesize::Int; stride=tilesize) where {D, TS}
        width, height = map(x -> size(data, x), (X,Y))
        xvals = 1:stride:width-TS+1
        yvals = 1:stride:height-TS+1
        tiles = [(x, y) for x in xvals for y in yvals]
        return new{D,TS}(data, tiles, tilesize)
    end
end

Base.length(x::TileSampler) = length(x.tiles)

Base.getindex(x::TileSampler, i::Int) = getindex(x, [i]) |> first
Base.getindex(x::TileSampler, i::AbstractVector) = _tiles(x, i)

Base.iterate(x::Apollo.TileSampler, state=1) = state > length(x) ? nothing : (x[state], state+1)

function Base.show(io::IO, x::TileSampler{D,TS}) where {D, TS}
    printstyled(io, "TileSampler(tile_size=$TS, num_tiles=$(length(x)))")
end

function _tiles(s::TileSampler{D,TS}, i::AbstractVector) where {D,TS}
    return [s.data[X(x:x+TS-1), Y(y:y+TS-1)] for (x, y) in s.tiles[i]]
end