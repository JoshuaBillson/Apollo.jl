struct Input{C}
    conv::C
end

function Input(channels::Int; batch_norm=true)
    return Input(ConvBlock((3,3), channels, 64, Flux.relu, batch_norm=batch_norm))
end

Flux.@layer Input

function (m::Input)(x)
    return m.conv(x)
end

struct DecoderBlock{U,C}
    up::U
    conv::C
end

Flux.@layer DecoderBlock

function DecoderBlock(up_channels, skip_channels, filters; batch_norm=false)
    return DecoderBlock(
        Flux.ConvTranspose((2,2), up_channels=>filters, stride=2), 
        ConvBlock((3,3), filters+skip_channels, filters, Flux.relu, batch_norm=batch_norm)
    )
end

function (m::DecoderBlock)(up, skip)
    return @pipe m.up(up) |> cat(_, skip, dims=3) |> m.conv
end

struct Decoder{D1,D2,D3,D4}
    decoder1::D1
    decoder2::D2
    decoder3::D3
    decoder4::D4
end

Flux.@layer Decoder

function Decoder(up_channels, skip_channels, filters; batch_norm=false)
    Decoder(
        DecoderBlock(up_channels[1], skip_channels[1], filters[1]; batch_norm=batch_norm), 
        DecoderBlock(up_channels[2], skip_channels[2], filters[2]; batch_norm=batch_norm), 
        DecoderBlock(up_channels[3], skip_channels[3], filters[3]; batch_norm=batch_norm), 
        DecoderBlock(up_channels[4], skip_channels[4], filters[4]; batch_norm=batch_norm), 
    )
end

function(m::Decoder)(x1, x2, x3, x4, x5)
    decoder_4 = m.decoder4(x5, x4)
    decoder_3 = m.decoder3(decoder_4, x3)
    decoder_2 = m.decoder2(decoder_3, x2)
    decoder_1 = m.decoder1(decoder_2, x1)
    return decoder_1
end

struct UNet{I,E,D,H}
    input::I
    encoder::E
    decoder::D
    head::H
end

Flux.@layer UNet

function UNet(input::I, encoder::E; nclasses=1, batch_norm=true) where {I,E}
    decoder_filters = [64, 128, 256, 512]
    encoder_filters = filters(E)
    skip_channels = encoder_filters[1:end-1]
    up_channels = vcat(decoder_filters[2:end], [encoder_filters[end]])
    return UNet(
        input, 
        encoder, 
        Decoder(up_channels, skip_channels, decoder_filters; batch_norm=batch_norm), 
        Flux.Conv((1,1), decoder_filters[1]=>nclasses)
    )
end

function (m::UNet)(x)
    s1, s2, s3, s4, s5 = m.input(x) |> m.encoder
    features = m.decoder(s1, s2, s3, s4, s5)
    out = m.head(features)
    return out
end

"""
function unet(in_features, n_classes; batch_norm=false)
    return UNet(
        UNetEncoder(in_features, 64, batch_norm, downsample=false),  # Encoder 1
        UNetEncoder(64, 128, batch_norm),                            # Encoder 2
        UNetEncoder(128, 256, batch_norm),                           # Encoder 3
        UNetEncoder(256, 512, batch_norm),                           # Encoder 4

        UNetDecoder(128, 64, batch_norm),                            # Decoder 1
        UNetDecoder(256, 128, batch_norm),                           # Decoder 2
        UNetDecoder(512, 256, batch_norm),                           # Decoder 3
        UNetDecoder(1024, 512, batch_norm),                          # Decoder 4

        UNetUp(128, 64),                                             # Up 1
        UNetUp(256, 128),                                            # Up 2
        UNetUp(512, 256),                                            # Up 3
        UNetUp(1024, 512),                                           # Up 4

        UNetBackbone(512, 1024, batch_norm),                         # Backbone

        Flux.Conv((1,1), 64=>n_classes)                              # Classification Head
    )
end

function unet(backbone::ResNet, n_classes; batch_norm=false)
    bfs = backbone.depth in [18, 34] ? [64, 64, 128, 256, 512] : [64, 256, 512, 1024, 2048]
    return UNet(
        backbone.input,                                 # Encoder 1
        backbone.backbone[1],                           # Encoder 2
        backbone.backbone[2],                           # Encoder 3
        backbone.backbone[3],                           # Encoder 4

        UNetDecoder(bfs[1]*2, bfs[1], batch_norm),      # Decoder 1
        UNetDecoder(bfs[2]*2, bfs[2], batch_norm),      # Decoder 2
        UNetDecoder(bfs[3]*2, bfs[3], batch_norm),      # Decoder 3
        UNetDecoder(bfs[4]*2, bfs[4], batch_norm),      # Decoder 4

        UNetUp(bfs[2], bfs[1]),                         # Up 1
        UNetUp(bfs[3], bfs[2]),                         # Up 2
        UNetUp(bfs[4], bfs[3]),                         # Up 3
        UNetUp(bfs[5], bfs[4]),                         # Up 4

        backbone.backbone[4],                           # Backbone

        Flux.Conv((1,1), 64=>n_classes)                 # Classification Head
    )
end

Flux.@layer UNet

function Apollo.activations(m::UNet, x)
    # Encoder Forward
    enc1 = m.encoder1(x)
    enc2 = m.encoder2(enc1)
    enc3 = m.encoder3(enc2)
    enc4 = m.encoder4(enc3)
    bottleneck = m.backbone(enc4)

    # Decoder Forward
    dec4 = @pipe m.up4(bottleneck) |> cat(_, enc4, dims=3) |> m.decoder4
    dec3 = @pipe m.up3(dec4) |> cat(_, enc3, dims=3) |> m.decoder3
    dec2 = @pipe m.up2(dec3) |> cat(_, enc2, dims=3) |> m.decoder2
    dec1 = @pipe m.up1(dec2) |> cat(_, enc1, dims=3) |> m.decoder1

    # Classification
    head = m.head(dec1)

    # Return Activations
    return (
        encoder_1=enc1, encoder_2=enc2, encoder_3=enc3, encoder_4=enc4,
        bottleneck=bottleneck, 
        decoder_4=dec4, decoder_3=dec3, decoder_2=dec2, decoder_1=dec1,
        prediction=head
    )
end

(m::UNet)(x::AbstractArray{<:Real,4}) = Apollo.activations(m, x).prediction
function (m::UNet)(x::AbstractDimArray)
    prediction = m(tensor(WHCN, x))
    newdims = (dims(x, X), dims(x, Y), Band(1:size(prediction,3)))
    return raster(prediction, WHCN, newdims)
end

function UNetEncoder(in_features::Int, out_features::Int, batch_norm; downsample=true)
    if downsample
        Flux.Chain(
            Flux.MaxPool((2,2)), 
            ConvBlock((3,3), in_features, out_features, Flux.relu, batch_norm=batch_norm)
        )
    else
        ConvBlock((3,3), in_features, out_features, Flux.relu, batch_norm=batch_norm)
    end

end

function UNetDecoder(in_features::Int, out_features::Int, batch_norm)
    ConvBlock((3,3), in_features, out_features, Flux.relu, batch_norm=batch_norm)
end

function UNetBackbone(in_features::Int, out_features::Int, batch_norm)
    Flux.Chain(
        Flux.MaxPool((2,2)), 
        ConvBlock((3,3), in_features, out_features, Flux.relu, batch_norm=batch_norm)
    )
end

function UNetUp(in_features::Int, out_features::Int)
    Flux.ConvTranspose((2,2), in_features=>out_features, stride=2)
end

"""