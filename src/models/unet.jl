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

struct UNet
    encoder1
    encoder2
    encoder3
    encoder4
    decoder1
    decoder2
    decoder3
    decoder4
    up1
    up2
    up3
    up4
    backbone
    head
end

function UNet(in_features, n_classes; batch_norm=false)
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

function UNet(backbone::ResNet, n_classes; batch_norm=false)
    return UNet(
        backbone.input,                        # Encoder 1
        backbone.backbone[1],                  # Encoder 2
        backbone.backbone[2],                  # Encoder 3
        backbone.backbone[3],                  # Encoder 4

        UNetDecoder(128, 64, batch_norm),      # Decoder 1
        UNetDecoder(512, 256, batch_norm),     # Decoder 2
        UNetDecoder(1024, 512, batch_norm),    # Decoder 3
        UNetDecoder(2048, 1024, batch_norm),   # Decoder 4

        UNetUp(256, 64),                       # Up 1
        UNetUp(512, 256),                      # Up 2
        UNetUp(1024, 512),                     # Up 3
        UNetUp(2048, 1024),                    # Up 4

        backbone.backbone[4],                  # Backbone

        Flux.Conv((1,1), 64=>n_classes)        # Classification Head
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