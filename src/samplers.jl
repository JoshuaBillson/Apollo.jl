struct TileSampler{T,TS,D}
    data::T
    tiles::Vector{Tuple{Int,Int}}
    tilesize::Int
    tshape::D

    function TileSampler(data::T, tilesize::Int; kwargs...) where {T}
        return TileSampler{T,tilesize,Nothing}(nothing, data, tilesize; kwargs...)
    end
    function TileSampler(::Type{D}, data::T, tilesize::Int; kwargs...) where {D<:TShape, T}
        return TileSampler{T,tilesize,D}(D, data, tilesize; kwargs...)
    end
    function TileSampler{T,TS,D}(::Any, data::T, tilesize::Int; stride=tilesize) where {D, T, TS}
        width, height = map(x -> size(data, x), (X,Y))
        xvals = 1:stride:width-TS+1
        yvals = 1:stride:height-TS+1
        tiles = [(x, y) for x in xvals for y in yvals]
        return new{T,TS,D}(data, tiles, tilesize, D())
    end
end

Base.length(x::TileSampler) = length(x.tiles)

Base.getindex(x::TileSampler{T,TS,Nothing}, i::Int) where{T,TS} = getindex(x, [i]) |> first
Base.getindex(x::TileSampler{T,TS,<:TShape}, i::Int) where{T,TS} = getindex(x, [i])
Base.getindex(x::TileSampler{T,TS,Nothing}, i::AbstractVector) where {T,TS} = _tiles(x, i)
Base.getindex(x::TileSampler{T,TS,D}, i::AbstractVector) where {T,TS,D<:TShape} = tensor(D, _tiles(x, i)...)

Base.iterate(x::Apollo.TileSampler, state=1) = state > length(x) ? nothing : (x[state], state+1)

function Base.show(io::IO, x::TileSampler{T,TS,D}) where {T, TS, D}
    printstyled(io, "TileSampler{$D}(tile_size=$TS, num_tiles=$(length(x)))")
end

function _tiles(s::TileSampler{T,TS}, i::AbstractVector) where {T,TS}
    return [s.data[X(x:x+TS-1), Y(y:y+TS-1)] for (x, y) in s.tiles[i]]
end