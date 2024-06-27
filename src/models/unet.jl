struct UnetUpBlock
    conv_transpose
    conv
end

function UnetUpBlock(up_features::Int, skip_features, batch_norm)
    conv_transpose = Flux.ConvTranspose((2,2), up_features=>skip_features, stride=2)
    conv = ConvBlock((3,3), up_features, skip_features, Flux.relu, batch_norm=batch_norm)
    return UnetUpBlock(conv_transpose, conv)
end

Flux.@functor UnetUpBlock

function (m::UnetUpBlock)(x1, x2)
    upsampled = m.conv_transpose(x1)
    concat = cat(upsampled, x2, dims=3)
    conv_out = m.conv(concat)
    return conv_out
end

struct UnetDownBlock
    conv
    downsample
end

function UnetDownBlock(in_features::Int, out_features::Int, batch_norm)
    conv = ConvBlock((3,3), in_features, out_features, Flux.relu, batch_norm=batch_norm)
    downsample = Flux.MaxPool((2,2))
    return UnetDownBlock(conv, downsample)
end

Flux.@functor UnetDownBlock

function (m::UnetDownBlock)(x)
    conv_out = m.conv(x)
    down = m.downsample(conv_out)
    return (conv_out, down)
end

struct UNet
    down1
    down2
    down3
    down4
    up1
    up2
    up3
    up4
    backbone
    head
end

function UNet(in_features, n_classes; batch_norm=false)
    down1 = UnetDownBlock(in_features, 64, batch_norm)
    down2 = UnetDownBlock(64, 128, batch_norm)
    down3 = UnetDownBlock(128, 256, batch_norm)
    down4 = UnetDownBlock(256, 512, batch_norm)

    up4 = UnetUpBlock(1024, 512, batch_norm)
    up3 = UnetUpBlock(512, 256, batch_norm)
    up2 = UnetUpBlock(256, 128, batch_norm)
    up1 = UnetUpBlock(128, 64, batch_norm)
        
    backbone = ConvBlock((3,3), 512, 1024, Flux.relu; batch_norm=batch_norm)
    
    head = Flux.Conv((1,1), 64=>n_classes)
        
    return UNet(down1, down2, down3, down4, up1, up2, up3, up4, backbone, head)
end

Flux.@functor(UNet)

function Flux.activations(m::UNet, x)
    skip1, down1 = m.down1(x)
    skip2, down2 = m.down2(down1)
    skip3, down3 = m.down3(down2)
    skip4, down4 = m.down4(down3)

    backbone = m.backbone(down4)

    up4 = m.up4(backbone, skip4)
    up3 = m.up3(up4, skip3)
    up2 = m.up2(up3, skip2)
    up1 = m.up1(up2, skip1)

    head = m.head(up1)

    return (
        encoder_1=skip1, encoder_2=skip2, encoder_3=skip3, encoder_4=skip4,
        bottleneck=backbone, 
        decoder_4=up4, decoder_3=up3, decoder_2=up2, decoder_1=up1,
        prediction=head
    )
end

(m::UNet)(x::AbstractArray{<:Real,4}) = m(Float32.(x))
(m::UNet)(x::AbstractArray{Float32,4}) = Flux.activations(m, x).prediction
function (m::UNet)(x::AbstractRaster)
    return @pipe tensor(x; dims=(X,Y,Band)) |> m(_) |> raster(_, (X,Y,Band), Rasters.dims(x))
end