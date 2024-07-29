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

struct DeeplabV3{I,E,A,D,H}
    input::I
    encoder::E
    aspp::A
    decoder1::D
    decoder2::D
    decoder3::D
    head::H
end

Flux.@layer DeeplabV3

function DeeplabV3(encoder::E; channels=3, nclasses=1) where {E <: AbstractEncoder}
    input = ConvBlock((3,3), channels, first(filters(encoder)), Flux.relu, batch_norm=true)
    return DeeplabV3(
        input,
        encoder,
        ASPP(filters(encoder)[4]), 
        Conv((1,1), filters(encoder)[2], 48, Flux.relu, batch_norm=true),
        Conv((3,3), 256+48, 256, Flux.relu, batch_norm=true),
        Conv((3,3), 256, 256, Flux.relu, batch_norm=true),
        Flux.Conv((1,1), 256=>nclasses),
    )
end

input(x::DeeplabV3) = x.input
encoder(x::DeeplabV3) = x.encoder
decoder(x::DeeplabV3) = (aspp=x.aspp, decoder1=x.decoder1, decoder2=x.decoder2, decoder3=x.decoder3)
head(x::DeeplabV3) = x.head

function (m::DeeplabV3)(x)
    # Encoder Forward
    encoder_out = m.input(x) |> m.encoder

    # ASPP Out
    aspp_out = encoder_out[4] |> m.aspp

    # Encoder Out
    input_a = Flux.upsample_bilinear(aspp_out, 4)
    input_b = encoder_out[2] |> m.decoder1
    concat = cat(input_a, input_b, dims=3)

    # Decoder Out
    decoder_out = @pipe m.decoder2(concat) |> m.decoder3(_) |> Flux.upsample_bilinear(_, 2)

    # Classification Head
    return m.head(decoder_out)
end