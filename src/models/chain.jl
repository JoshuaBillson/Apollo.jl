struct Chain{T,I,O}
    layers::T
end

function Chain(layers...; in=(X,Y,Band), out=(X,Y,Band))
    Chain{typeof(layers),name(in),name(out)}(layers)
end

Flux.@functor Chain

activations(c::Chain, input) = _extraChain(Tuple(c.layers), input)

(c::Chain)(x::AbstractArray{<:Real}) = c(Float32.(x))
(c::Chain)(x::AbstractArray{<:Float32}) = last(activations(c, x))
function (c::Chain{T,I,O})(x::AbstractRaster{<:Real,N}) where {T,I,O,N}
    @pipe tensor(x, dims=_name_to_dim.(I)) |> c |> raster(_, _name_to_dim.(O), Rasters.dims(x))
end

_extraChain(::Tuple{}, x) = ()
function _extraChain(fs::Tuple, x)
    res = first(fs)(x)
    return (res, _extraChain(Base.tail(fs), res)...)
end