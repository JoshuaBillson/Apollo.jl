"""
    DeeplabV3(;encoder=ResNet(depth=34, weights=:ImageNet), inchannels=3, nclasses=1)

Construct a DeeplabV3+ model.
"""
struct DeeplabV3{E,A,D,H}
    encoder::E
    aspp::A
    decoder::D
    head::H
end

Flux.@layer :expand DeeplabV3

function DeeplabV3(;encoder=ResNet(depth=34, weights=:ImageNet), inchannels=3, nclasses=1)
    return DeeplabV3(
        build_encoder(encoder, inchannels)[1:4], 
        build_deeplab_aspp(filters(encoder)[4]),
        build_deeplab_decoder(filters(encoder)[2], 256), 
        Flux.Chain(Conv(3, 256, 256, Flux.relu), Flux.Conv((1,1), 256=>nclasses))
    )
end

function deeplabv3(;encoder=ResNet(depth=34, weights=:ImageNet), inchannels=3, nclasses=1)
    _filters = filters(encoder)
    encoder1, encoder2, encoder3, encoder4, _ = build_encoder(encoder, inchannels)
    aspp = build_deeplab_aspp(_filters[4])
    Flux.Chain(
        Flux.Chain(;encoder1, encoder2), 
        Flux.Parallel(
            (a, b) -> cat(a, b, dims=3), 
            Flux.Chain(;encoder3, encoder4, aspp, up=Base.Fix2(Flux.upsample_bilinear, (4,4))), 
            Conv(1, _filters[2], 48, Flux.relu),
        ), 
        Flux.Chain(
            Conv(3, 256+48, 256, Flux.relu),
            Conv(3, 256, 256, Flux.relu),
            Base.Fix2(Flux.upsample_bilinear, (4,4))
        ), 
        Flux.Chain(
            Conv(3, 256, 256, Flux.relu), 
            Flux.Conv((1,1), 256=>nclasses)
        )
    )
end


function (m::DeeplabV3)(x)
    # Encoder Forward
    x1 = m.encoder[1](x)
    x2 = m.encoder[2](x1)
    x3 = m.encoder[3](x2)
    x4 = m.encoder[4](x3)

    # ASPP Out
    aspp_out = m.aspp(x4)

    # Encoder Out
    input_a = Flux.upsample_bilinear(aspp_out, 4)
    input_b = m.decoder[1](x2)
    concat = cat(input_a, input_b, dims=3)

    # Decoder Out
    decoder_out = @pipe m.decoder[2](concat) |> m.decoder[3](_) |> Flux.upsample_bilinear(_, 4)

    # Classification Head
    return m.head(decoder_out)
end

function build_deeplab_aspp(in_filters)
    return Flux.Chain(
        Flux.Parallel(
            aspp_concat,
            Flux.Chain(Flux.GlobalMeanPool(), Conv(1, in_filters, 256, Flux.relu, dilation=1)), 
            Conv(1, in_filters, 256, Flux.relu, dilation=1), 
            Conv(3, in_filters, 256, Flux.relu, dilation=6), 
            Conv(3, in_filters, 256, Flux.relu, dilation=12), 
            Conv(3, in_filters, 256, Flux.relu, dilation=18), 
        ), 
        Conv(1, 1280, 256, Flux.relu)
    )
end

function aspp_concat(x1, x2, x3, x4, x5)
    x1 = Flux.upsample_bilinear(x1, size=size(x2)[1:2])
    return cat(x1, x2, x3, x4, x5, dims=3)
end

function build_deeplab_decoder(low_level_filters, aspp_filters)
    return (
        Conv(1, low_level_filters, 48, Flux.relu),
        Conv(3, aspp_filters+48, 256, Flux.relu, batch_norm=true),
        Conv(3, 256, 256, Flux.relu),
    )
end