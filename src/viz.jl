"""
    linear_stretch(x::AbstractArray{<:Real,3}, lb::Vector{<:Real}, ub::Vector{<:Real})

Perform a linear histogram stretch on `x` such that `lb` is mapped to 0 and `ub` is mapped to 1.
Values outside the interval `[lb, ub]` will be clamped.
"""
linear_stretch(x::AbstractArray{<:Real,4}, lb, ub) = linear_stretch(rmobs(x), lb, ub)
function linear_stretch(x::AbstractArray{<:Real,3}, lb::Vector{<:Real}, ub::Vector{<:Real})
    lb = reshape(lb, (1,1,:))
    ub = reshape(ub, (1,1,:))
    return clamp!((x .- lb) ./ (ub .- lb), 0, 1)
end

"""
    rgb(x, lb, ub; bands=[1,2,3])

Visualze the specified bands of `x` as an rgb image.

# Parameters
- `x`: The `AbstractArray` to be visualized. Must contain at least 3 bands.
- `lb`: A vector of lower bounds for each channel in `x`. Used by `linear_stretch`.
- `ub`: A vector of upper bounds for each channel in `x`. Used by `linear_stretch`.
- `bands`: A vector of band indices to assign to red, green, and blue, respectively.
"""
rgb(x::AbstractRaster, lb, ub; kwargs...) = rgb(x[X(:),Y(:),Band(:)].data, lb, ub; kwargs...)
rgb(x::AbstractArray{<:Real,4}, lb, ub; kwargs...) = rgb(rmobs(x), lb, ub; kwargs...)
function rgb(x::AbstractArray{<:Real,3}, lb, ub; bands=[1,2,3])
    @pipe linear_stretch(x, lb, ub)[:,:,bands] |>
    ImageCore.N0f8.(_) |>
    permutedims(_, (3, 2, 1)) |>
    ImageCore.colorview(ImageCore.RGB, _)
end

"""
    binmask(x)

Visualze `x` as a black/white binary mask.
"""
binmask(x::AbstractRaster) = binmask(x[X(:),Y(:)].data)
binmask(x::AbstractArray{<:Real,4}) = binmask(reshape(x, size(x)[1:2]))
binmask(x::AbstractArray{<:Real,3}) = binmask(reshape(x, size(x)[1:2]))
function binmask(x::AbstractArray{<:Real,2})
    @pipe ImageCore.N0f8.(x) |>
    permutedims(_, (2, 1)) |>
    ImageCore.Gray.(_)
end

"""
    mosaicview(size::Tuple{Int,Int}, imgs...)

Plot a mosaic of images with size (rows, cols).
"""
function mosaicview(size::Tuple{Int,Int}, imgs...)
    rows, cols = size
    MosaicViews.mosaicview(imgs..., nrow=rows, ncols=cols, fillvalue=1.0, rowmajor=true, npad=5)
end