abstract type TShape end
struct WHCN <: TShape end
struct WHCLN <: TShape end

tensor(xs::AbstractVector; kwargs...) = tensor(xs...; kwargs...)
tensor(x::AbstractArray; precision=:f32, kwargs...) = _precision(x, precision)
function tensor(xs...; kwargs...) 
    tensors = map(x -> tensor(x; kwargs...), xs)
    if any(!=(size(first(tensors))), map(size, tensors))
        throw(ArgumentError("Cannot stack tensors of different size!"))
    end
    return _stack(tensors...)
end
function tensor(x::AbstractDimStack; precision=:f32, layerdim=Band)
    return tensor(catlayers(x, layerdim), precision=precision)
end
function tensor(x::AbstractDimArray; precision=:f32, layerdim=Band)
    if !hasdim(x, Band)
        return tensor(add_dim(x, Band); precision=precision)
    else
        _dims = Rasters.commondims((X,Y,Z,Band,Ti), Rasters.dims(x))
        return @pipe _permute(x, _dims).data |> _unsqueeze |> _precision(_, precision)
    end
end