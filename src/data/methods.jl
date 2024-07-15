function resample(x::HasDims, scale, method=:bilinear)
    _check_resample_method(method)
    newsize = (size(x,X), size(x,Y)) .* scale
    return Rasters.resample(x, size=newsize, method=method)
end

function upsample(x::AbstractArray, scale, method=:bilinear)
    @match method begin
        :linear => Flux.upsample_linear(x, scale)
        :bilinear => Flux.upsample_bilinear(x, scale)
        :trilinear => Flux.upsample_trilinear(x, scale)
        :nearest => Flux.upsample_nearest(x, scale)
        _ => throw(ArgumentError("`method` must be one of :linear, :bilinear, :trilinear, or :nearest!"))
    end
end

function resize(x::HasDims, newsize, method=:bilinear)
    _check_resample_method(method)
    return Rasters.resample(x, size=newsize, method=method)
end

function crop(x::HasDims, cropsize)
end

function _check_resample_method(method)
    valid_methods = [:near, :bilinear, :cubic, :cubicspline, :lanczos, :average]
    if !(method in valid_methods)
        throw(ArgumentError("`method` must be one of $(join(map(x -> ":$x", valid_methods), ", ", ", or "))!"))
    end
end