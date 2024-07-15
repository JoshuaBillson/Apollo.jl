function stats(x::AbstractArray)
    data = _extract_bands(x)
    return (map(mean, data), map(std, data))
end
function stats(xs)
    ms, μ, σ = (0.0, 0.0, 0.0)
    for x in xs
        # Extract Batch Data
        d = _extract_bands(x)

        # Compute Batch Statistics
        n = map(length, d)
        μ_n = map(mean, d)
        σ_n = map(var, d)

        # Copy Old Statistics
        m = deepcopy(ms)
        μ_m = deepcopy(μ)
        σ_m = deepcopy(σ)

        # Update Statistics
        μ = @. (m / (m + n)) * μ_m + (n / (m + n)) * μ_n
        σ = @. (m / (m+n)) * σ_m + (n / (m+n)) * σ_n + (m*n) / (m+n)^2 * (μ_m-μ_n)^2
        ms = @. ms + n
    end
    return μ, sqrt.(σ)
end

function normalize(x::AbstractArray{<:T,N}, μ::AbstractVector, σ::AbstractVector; dim=3) where {T <: AbstractFloat, N}
    shape = Tuple(ifelse(i == dim, :, 1) for i in 1:N)
    μ = T.(reshape(μ, shape))
    σ = T.(reshape(σ, shape))
    return (x .- μ) ./ σ
end
function normalize(x::AbstractRaster, μ::AbstractVector, σ::AbstractVector; dim=dimnum(x, Band))
    return Rasters.modify(x -> normalize(x, μ, σ, dim=dim), x)
end

function denormalize(x::AbstractArray{<:T,N}, μ::AbstractVector, σ::AbstractVector; dim=3) where {T <: AbstractFloat, N}
    shape = Tuple(ifelse(i == dim, :, 1) for i in 1:N)
    μ = T.(reshape(μ, shape))
    σ = T.(reshape(σ, shape))
    return (x .* σ) .+ μ
end
function denormalize(x::AbstractRaster, μ::AbstractVector, σ::AbstractVector; dim=dimnum(x, Band))
    return Rasters.modify(x -> denormalize(x, μ, σ, dim=dim), x)
end

_extract_bands(x::AbstractArray{<:Real}) =  _extract_bands(Float64.(x))
_extract_bands(x::AbstractArray{Float64}) = [vec(selectdim(x, 3, i)) for i in 1:_nbands(x)]
function _extract_bands(x::AbstractDimArray)
    bands = [x[Band(i)].data for i in 1:_nbands(x)]
    map(bands) do band
        @pipe ifelse.(band .=== missingval(x), missing, band) |> 
        skipmissing(_) |> 
        collect(_) |>
        Float64.(_)
    end
end

_nbands(x::AbstractDimArray) = size(x, Band)
_nbands(x::AbstractArray) = size(x, 3)