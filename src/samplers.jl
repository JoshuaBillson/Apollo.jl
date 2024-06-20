struct TileSampler{D,T}
	stack::D
	tiles::Vector{Tuple{Int,Int}}
	tilesize::Int

	function TileSampler(stack::D, tilesize::Int, stride::Int) where {D}
		width, height = size(stack)
		xvals = 1:stride:width-tilesize+1
		yvals = 1:stride:height-tilesize+1
		tiles = [(x, y) for x in xvals for y in yvals]
		return new{D,tilesize}(stack, tiles, tilesize)
	end
end

Base.length(x::TileSampler) = length(x.tiles);

function Base.getindex(x::TileSampler, i::Int)
	return Base.getindex(x, [i]) |> first
end;

function Base.getindex(x::TileSampler, i::AbstractVector)
	stack = x.stack
	tiles = x.tiles[i]
	tilespan = x.tilesize - 1
	return [stack[X(x:x+tilespan), Y(y:y+tilespan)] for (x, y) in tiles]
end