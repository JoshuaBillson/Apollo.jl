struct UnetUpBlock
    conv_transpose
    conv1
    conv2
end

function UnetUpBlock(features::Int)
    conv_transpose = Flux.ConvTranspose((2,2), features*2=>features, stride=2)
    conv1 = Flux.Conv((3,3), features*2=>features, Flux.relu, pad=Flux.SamePad())
    conv2 = Flux.Conv((3,3), features=>features, Flux.relu, pad=Flux.SamePad())
    return UnetUpBlock(conv_transpose, conv1, conv2)
end

Flux.@functor UnetUpBlock

function (m::UnetUpBlock)(x1, x2)
    upsampled = m.conv_transpose(x1)
    concat = cat(upsampled, x2, dims=3)
    conv1_out = m.conv1(concat)
    conv2_out = m.conv2(conv1_out)
    return conv2_out
end

struct UnetDownBlock
    conv1
    conv2
    downsample
end

function UnetDownBlock(in_features::Int, out_features::Int)
    conv1 = Flux.Conv((3,3), in_features=>out_features, Flux.relu, pad=Flux.SamePad())
    conv2 = Flux.Conv((3,3), out_features=>out_features, Flux.relu, pad=Flux.SamePad())
    downsample = Flux.MaxPool((2,2))
    return UnetDownBlock(conv1, conv2, downsample)
end

Flux.@functor UnetDownBlock

function (m::UnetDownBlock)(x)
    conv1_out = m.conv1(x)
    conv2_out = m.conv2(conv1_out)
    down = m.downsample(conv2_out)
    return (conv2_out, down)
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

function UNet(in_features, n_classes)
    down1 = UnetDownBlock(in_features, 64)
    down2 = UnetDownBlock(64, 128)
    down3 = UnetDownBlock(128, 256)
    down4 = UnetDownBlock(256, 512)

    up4 = UnetUpBlock(512)
    up3 = UnetUpBlock(256)
    up2 = UnetUpBlock(128)
    up1 = UnetUpBlock(64)
        
    backbone = Flux.Chain(
        Flux.Conv((3,3), 512=>1024, Flux.relu, pad=Flux.SamePad()), 
        Flux.Conv((3,3), 1024=>1024, Flux.relu, pad=Flux.SamePad()) )
    
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

    head = Flux.softmax(m.head(up1), dims=3)

    return (encoder=[skip1, skip2, skip3, skip4], backbone=backbone, decoder=[up4, up3, up2, up1], prediction=head)
end

(m::UNet)(x::Array{<:Real,2}) = m(MLUtils.unsqueeze(x, 3))
(m::UNet)(x::Array{<:Real,3}) = m(MLUtils.unsqueeze(x, 4))
(m::UNet)(x::Array{<:Real,4}) = m(Float32.(x))
(m::UNet)(x::Array{Float32,4}) = Flux.activations(m, x).prediction
(m::UNet)(x::AbstractRasterStack) = m(Raster(x))
function (m::UNet)(x::T) where {T <: AbstractRaster}
    return Raster(m(x.data)[:,:,:,1], (dims(x)[1:2]..., Band), missingval=missingval(x))
end