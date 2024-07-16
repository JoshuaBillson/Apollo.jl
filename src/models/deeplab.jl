function ASPP_Block(kernel::Int, in_filters::Int, out_filters::Int, dilation::Int=1)
    Flux.Chain(
        Flux.Conv((kernel, kernel), in_filters=>out_filters, pad=Flux.SamePad(), dilation=dilation), 
        Flux.BatchNorm(out_filters, Flux.relu) 
    )
end

struct ASPP{P,O1,O6,O12,O18,O}
    pool::P
    out_1::O1
    out_6::O6
    out_12::O12
    out_18::O18
    out::O
end

Flux.@layer ASPP

function ASPP(in_filters)
    return ASPP(
        Flux.Chain(Flux.GlobalMeanPool(), ASPP_Block(1, in_filters, 256, 1)), 
        ASPP_Block(1, in_filters, 256, 1), 
        ASPP_Block(3, in_filters, 256, 6), 
        ASPP_Block(3, in_filters, 256, 12), 
        ASPP_Block(3, in_filters, 256, 18), 
        ASPP_Block(1, 1280, 256, 1)
    )
end

function (m::ASPP)(x)
    pool_out = @pipe m.pool(x) |> Flux.upsample_bilinear(_, size=size(x)[1:2])
    out_1 = m.out_1(x)
    out_6 = m.out_6(x)
    out_12 = m.out_12(x)
    out_18 = m.out_18(x)
    concat = cat(pool_out, out_1, out_6, out_12, out_18, dims=3)
    out = m.out(concat)
    return out
end

struct DeeplabV3{I,B,A,D,H}
    input::I
    backbone::B
    aspp::A
    decoder1::D
    decoder2::D
    decoder3::D
    head::H
end

Flux.@layer DeeplabV3

function DeeplabV3(in_features, nclasses; depth=50, pretrain=false)
    backbone = ResNet(depth, pretrain=pretrain, channels=in_features)
    return DeeplabV3(
        backbone.input,
        backbone.backbone[1:3],
        ASPP(1024), 
        Flux.Chain(Flux.Conv((1,1), 256=>48), Flux.BatchNorm(48, Flux.relu)), 
        Flux.Chain(Flux.Conv((3,3), 304=>256, pad=Flux.SamePad()), Flux.BatchNorm(256, Flux.relu)), 
        Flux.Chain(Flux.Conv((3,3), 256=>256, pad=Flux.SamePad()), Flux.BatchNorm(256, Flux.relu)), 
        Flux.Conv((1,1), 256=>nclasses),
    )
end

function (m::DeeplabV3)(x)
    # Backbone Forward
    input_out = m.input(x)
    backbone_out = Flux.activations(m.backbone, input_out)

    # ASPP Out
    aspp_out = last(backbone_out) |> m.aspp

    # Encoder Out
    input_a = Flux.upsample_bilinear(aspp_out, 4)
    input_b = first(backbone_out) |> m.decoder1
    concat = cat(input_a, input_b, dims=3)

    # Decoder Out
    decoder_out = @pipe m.decoder2(concat) |> m.decoder3(_) |> Flux.upsample_bilinear(_, 4)

    # Classification Head
    return m.head(decoder_out)
end