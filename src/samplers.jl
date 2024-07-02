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
		xvals = 1:stride:width-tilesize+1
		yvals = 1:stride:height-tilesize+1
		tiles = [(x, y) for x in xvals for y in yvals]
		return new{T,TS,D}(data, tiles, tilesize, D())
	end
end

struct TileSet{T,D}
    data::T
    tshape::D

    function TileSet(::Type{D}, data::T) where {D <: TShape, T}
        return new{T,D}(data, D())
    end
end

Base.length(x::TileSet) = length(x.data)

Base.getindex(x::TileSet, i::Int) = getindex(x, [i])
function Base.getindex(x::TileSet{T,D}, i::AbstractVector) where {T,D}
    tensor(D, x.data[i]...)
end

Base.length(x::TileSampler) = length(x.tiles);

Base.getindex(x::TileSampler{T,TS,Nothing}, i::Int) where{T,TS} = getindex(x, [i]) |> first
Base.getindex(x::TileSampler{T,TS,<:TShape}, i::Int) where{T,TS} = getindex(x, [i])
function Base.getindex(x::TileSampler{T,TS,Nothing}, i::AbstractVector) where {T,TS}
    return _tiles(x, i)
end
function Base.getindex(x::TileSampler{T,TS,D}, i::AbstractVector) where {T,TS,D<:TShape}
    return tensor(D, _tiles(x, i)...)
end

function _tiles(x::TileSampler{T,TS,D}, i::AbstractVector) where {T,TS,D}
    data = x.data
	tiles = x.tiles[i]
	tilespan = TS - 1
    return [data[X(x:x+tilespan), Y(y:y+tilespan)] for (x, y) in tiles]
end

Base.iterate(x::Apollo.TileSampler, state=1) = state > length(x) ? nothing : (x[state], state+1)

function Base.show(io::IO, x::TileSampler{T,TS,D}) where {T, TS, D}
    printstyled(io, "TileSampler{$D}(tile_size=$TS, num_tiles=$(length(x)))")
end