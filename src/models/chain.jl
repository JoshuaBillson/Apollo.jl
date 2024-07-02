struct Chain{T,I,O}
    layers::T
end

function Chain(layers...)
    Chain{typeof(layers),WHCN,WHCN}(layers)
end

function Chain(shape::Tuple, layers...)
    Chain{typeof(layers),shape,shape}(layers)
end

function Chain(shape::Pair, layers...)
    Chain{typeof(layers),first(shape),last(shape)}(layers)
end

Flux.@functor Chain

activations(c::Chain, input) = _extraChain(Tuple(c.layers), input)

(c::Chain)(x::AbstractArray{<:AbstractFloat}) = last(activations(c, x))
function (c::Chain{T,I,O})(x::HasDims) where {T,I,O}
    @pipe tensor(I, x) |> c |> raster(_, O, dims(x))
end

_extraChain(::Tuple{}, x) = ()
function _extraChain(fs::Tuple, x)
    res = first(fs)(x)
    return (res, _extraChain(Base.tail(fs), res)...)
end