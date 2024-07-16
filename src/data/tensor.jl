# Precision Defaults to Float32
tensor(args...; kwargs...) = tensor(Float32, args...; kwargs...)

# Vector of Observations
tensor(T::Type{<:AbstractFloat}, xs::AbstractVector; kwargs...) = tensor(T, xs...; kwargs...)

# Vararg Observations
tensor(T::Type{<:AbstractFloat}, xs...; kwargs...) = _stack(map(x -> tensor(T, x; kwargs...), xs)...)

# AbstractRasterStack
tensor(T::Type{<:AbstractFloat}, x::AbstractRasterStack; layerdim=Band) = tensor(T, catlayers(x, layerdim))

# AbstractRaster
function tensor(T::Type{<:AbstractFloat}, x::AbstractRaster; kwargs...)
    if !hasdim(x, Band)
        return tensor(T, add_dim(x, Band); precision=precision)
    end
    _dims = Rasters.commondims((X,Y,Z,Band,Ti), Rasters.dims(x))
    return tensor(T, _permute(x, _dims).data)
end

# Base Method
tensor(T::Type{<:AbstractFloat}, x::AbstractArray; kwargs...) = _precision(T, putobs(x))