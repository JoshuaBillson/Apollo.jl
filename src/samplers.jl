struct TileSampler{D,T}
	stack::D
	tiles::Vector{Tuple{Int,Int}}
	tilesize::Int

	function TileSampler(stack::D, tilesize::Int; stride=tilesize) where {D<:HasDims}
		width, height = map(x -> size(stack, x), (X,Y))
		xvals = 1:stride:width-tilesize+1
		yvals = 1:stride:height-tilesize+1
		tiles = [(x, y) for x in xvals for y in yvals]
		return new{D,tilesize}(stack, tiles, tilesize)
	end
end

Base.length(x::TileSampler) = length(x.tiles);

function Base.getindex(x::TileSampler, i::Int)
	return Base.getindex(x, [i]) |> first
end

function Base.getindex(x::TileSampler, i::AbstractVector)
	stack = x.stack
	tiles = x.tiles[i]
	tilespan = x.tilesize - 1
    return [stack[X(x:x+tilespan), Y(y:y+tilespan)] for (x, y) in tiles]
end

struct Pipeline{D,T}
    dims::D
    sampler::T
    Pipeline(dims::D, sampler::T) where {D,T} = new{D,T}(dims, sampler)
end

function Pipeline(samplers::Vararg{Pair})
    _samplers = map(first, samplers)
    _dims = map(last, samplers)
    Pipeline(_dims, _samplers)
end

function Pipeline(samplers::Pair)
    Pipeline(last(samplers), first(samplers))
end

Base.length(x::Pipeline) = length(x.sampler)

Base.length(x::Pipeline{<:Tuple,<:Tuple}) = x.sampler |> first |> length

function Base.getindex(x::Pipeline, i::Int)
    return @pipe getindex(x.sampler, i) |> tensor(_; dims=x.dims)
end

function Base.getindex(x::Pipeline, i::AbstractVector)
    return @pipe getindex(x.sampler, i) |> map(tile -> tensor(tile; dims=x.dims), _) |> _stack(_...)
end

function Base.getindex(x::Pipeline{<:Tuple,<:Tuple}, i::Int)
    tiles = map(x -> getindex(x, i), x.sampler)
    return Tuple(tensor(t; dims=d) for (t, d) in zip(tiles, x.dims))
end

function Base.getindex(x::Pipeline{<:Tuple,<:Tuple}, i::AbstractVector)
    tiles = map(x -> getindex(x, i), x.sampler)
    return Tuple(tensor(t...; dims=d) for (t, d) in zip(tiles, x.dims))
end

function Base.show(io::IO, x::TileSampler)
    printstyled(io, "TileSampler(size=$(x.tilesize), length=$(length(x)))")
end