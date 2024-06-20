function SSC_Conv(in_filters, out_filters)
	return Flux.Conv((3, 3), in_filters=>out_filters, Flux.leakyrelu, pad=Flux.SamePad())
end

function SSC_BottleNeck()
	return Flux.Chain( SSC_Conv(128, 128), SSC_Conv(128, 128) )
end

struct SSC_CNN
    enc1
    enc2
    enc3
    dec3
    dec2
    dec1
    skip1
    skip2
    skip3
    bottleneck
    head
end

function SSC_CNN()
    return SSC_CNN(
        SSC_Conv(10, 128), SSC_Conv(128, 128), SSC_Conv(128, 128),   # Encoder
        SSC_Conv(192, 128), SSC_Conv(192, 128), SSC_Conv(192, 128),  # Decoder
        SSC_Conv(128, 64), SSC_Conv(128, 64), SSC_Conv(128, 64),     # Skip
        Flux.Chain(SSC_Conv(128, 128), SSC_Conv(128, 128)),          # Bottleneck
        Flux.Conv((3,3), 128=>6, pad=Flux.SamePad()))                # Head
end

Flux.@functor(SSC_CNN)

(m::SSC_CNN)(lr::Array{<:Real,3}, hr::Array{<:Real,3}) = m(MLUtils.unsqueeze(lr, 4), MLUtils.unsqueeze(hr, 4))
(m::SSC_CNN)(lr::Array{<:Real,4}, hr::Array{<:Real,4}) = m(Float32.(lr), Float32.(hr))
function (m::SSC_CNN)(lr::Array{Float32,4}, hr::Array{Float32,4})
    # Upsample LR Bands
    lr_up = Flux.upsample_bilinear(lr, (2, 2))
    x = cat(lr_up, hr, dims=3)

    # Encoder Forward
    e1 = m.enc1(x)
    e2 = m.enc2(e1)
    e3 = m.enc3(e2)

    # Skip Forward
    s1 = m.skip1(e1)
    s2 = m.skip2(e2)
    s3 = m.skip3(e3)

    # Backbone Forward
    bottleneck = m.bottleneck(e3)

    # Decoder Forward
    d3 = m.dec3(cat(bottleneck, s3, dims=3))
    d2 = m.dec2(cat(d3, s2, dims=3))
    d1 = m.dec1(cat(d2, s1, dims=3))

    # Residual Out
    residual = m.head(d1)
    return residual .+ lr_up
end

function (m::SSC_CNN)(lr::AbstractRaster{<:Real,3}, hr::AbstractRaster{<:Real,3})
    _dims = (dims(hr)[1:2]..., dims(lr, Band))
    return Raster(m(lr.data, hr.data)[:,:,:,1], _dims, missingval=missingval(lr))
end

function (m::SSC_CNN)(lr::AbstractRasterStack, hr::AbstractRasterStack)
    return RasterStack(m(Raster(lr), Raster(hr)), layersfrom=Band)
end