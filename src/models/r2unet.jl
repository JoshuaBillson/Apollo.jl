struct RecurrentConv{C,T}
    conv::C
end

Flux.@layer RecurrentConv

RecurrentConv(conv::C, t::Int) where{C} = RecurrentConv{C,t}(conv)
function RecurrentConv(kernel::Tuple, filters::Int; t=2, batch_norm=false)
    if batch_norm
        RecurrentConv(
            Flux.Chain(
                Flux.Conv(kernel, filters=>filters, pad=Flux.SamePad()),
                Flux.BatchNorm(filters), 
                Flux.relu
            ), t
        )
    else
        RecurrentConv(Flux.Conv(kernel, filters=>filters, Flux.relu, pad=Flux.SamePad()), t)
    end
end

function (m::RecurrentConv{C,T})(x) where {C,T}
    x1 = nothing
    for i in 1:T
        if i == 1
            x1 = m.conv(x)
        else
            x1 = m.conv(x .+ x1)
        end
    end
    return x1
end

struct RCNN_Block{C1,C2}
    conv1::C1
    conv2::C2
end

Flux.@layer RCNN_Block

function RCNN_Block(kernel::Tuple, in::Int, out::Int; t=2, batch_norm=false)
    conv1 = Flux.Conv((1,1), in=>out)
    conv2 = Flux.Chain(
        RecurrentConv(kernel, out, t=t, batch_norm=batch_norm), 
        RecurrentConv(kernel, out, t=t, batch_norm=batch_norm))
    return RCNN_Block(conv1, conv2)
end

function (m::RCNN_Block)(x)
    x1 = m.conv1(x)
    x2 = m.conv2(x1)
    return x1 .+ x2
end

function R2UNetBlock(in_features::Int, out_features::Int, t::Int, batch_norm)
    RCNN_Block((3,3), in_features, out_features, t=t, batch_norm=batch_norm)
end

function R2UNetUp(in_features::Int, out_features::Int)
    Flux.ConvTranspose((2,2), in_features=>out_features, stride=2)
end

struct R2UNet{E,D,U,B,H}
    encoder1::E
    encoder2::E
    encoder3::E
    encoder4::E
    decoder1::D
    decoder2::D
    decoder3::D
    decoder4::D
    up1::U
    up2::U
    up3::U
    up4::U
    backbone::B
    head::H
end

function R2UNet(in_features, n_classes; t=2, batch_norm=false)
    return R2UNet(
        R2UNetBlock(in_features, 64, t, batch_norm),  # Encoder 1
        R2UNetBlock(64, 128, t, batch_norm),          # Encoder 2
        R2UNetBlock(128, 256, t, batch_norm),         # Encoder 3
        R2UNetBlock(256, 512, t, batch_norm),         # Encoder 4

        R2UNetBlock(128, 64, t, batch_norm),          # Decoder 1
        R2UNetBlock(256, 128, t, batch_norm),         # Decoder 2
        R2UNetBlock(512, 256, t, batch_norm),         # Decoder 3
        R2UNetBlock(1024, 512, t, batch_norm),        # Decoder 4

        R2UNetUp(128, 64),                            # Up 1
        R2UNetUp(256, 128),                           # Up 2
        R2UNetUp(512, 256),                           # Up 3
        R2UNetUp(1024, 512),                          # Up 4

        R2UNetBlock(512, 1024, t, batch_norm),        # Backbone

        Flux.Conv((1,1), 64=>n_classes)               # Classification Head
    )
end

Flux.@layer R2UNet

function Flux.activations(m::R2UNet, x)
    # Encoder Forward
    enc1 = m.encoder1(x)
    enc2 = Flux.maxpool(enc1, (2,2)) |> m.encoder2
    enc3 = Flux.maxpool(enc2, (2,2)) |> m.encoder3
    enc4 = Flux.maxpool(enc3, (2,2)) |> m.encoder4
    bottleneck = Flux.maxpool(enc4, (2,2)) |> m.backbone

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

(m::R2UNet)(x::AbstractArray{<:Real,4}) = Flux.activations(m, x).prediction