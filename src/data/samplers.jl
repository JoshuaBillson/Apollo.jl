struct TileSampler{T<:DType,TS,D<:HasDims}
    data::D
    dtype::T
    tiles::Vector{Tuple{Int,Int}}
    tilesize::Int

    function TileSampler(data::D, dtype::T, tilesize::Int; kwargs...) where {D<:HasDims,T<:DType}
        return TileSampler{T,tilesize,D}(data, dtype, tilesize; kwargs...)
    end
    function TileSampler{T,TS,D}(data::D, dtype::T, tilesize::Int; stride=tilesize) where {T<:DType,TS,D<:HasDims}
        width, height = map(x -> size(data, x), (X,Y))
        xvals = 1:stride:width-TS+1
        yvals = 1:stride:height-TS+1
        tiles = [(x, y) for x in xvals for y in yvals]
        return new{T,TS,D}(data, dtype, tiles, tilesize)
    end
end

is_tile_source(::TileSampler) = true
dtype(x::TileSampler{T}) where {T<:DType} = x.dtype

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

struct TileSeq{T<:DType,D} <: TileSource
    dtype::T
    tiles::D
end

function TileSeq(tiles::D, dtype::T) where {D<:AbstractVector, T<:DType}
    return TileSeq{T,D}(dtype, tiles)
end

function _tiles(s::TileSampler{D,TS}, i::AbstractVector) where {D,TS}
    return [s.data[X(x:x+TS-1), Y(y:y+TS-1)] for (x, y) in s.tiles[i]]
end