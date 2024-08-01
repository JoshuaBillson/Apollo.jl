"""
    SSC_CNN()

Construct an SSC_CNN model ([Nguyen et al.](https://doi.org/10.1109/IGARSS39084.2020.9323614)) 
to sharpen the 20m Sentinel-2 bands to a resolution of 10m.
"""
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

Flux.@layer SSC_CNN

function (m::SSC_CNN)(lr::AbstractArray{Float32,4}, hr::AbstractArray{Float32,4})
    # Upsample LR Bands
    #lr_up = Flux.upsample_bilinear(lr, (2, 2))
    x = cat(lr, hr, dims=3)

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
    return residual .+ lr
end

function SSC_Conv(in_filters, out_filters)
	return Flux.Conv((3, 3), in_filters=>out_filters, Flux.leakyrelu, pad=Flux.SamePad())
end

function SSC_BottleNeck()
	return Flux.Chain( SSC_Conv(128, 128), SSC_Conv(128, 128) )
end
