struct UNet{I,E,D,H}
    input::I
    encoder::E
    decoder::D
    head::H
end

Flux.@layer UNet

UNet(; batch_norm=true, kwargs...) = UNet(StandardEncoder(batch_norm=batch_norm); kwargs...)
function UNet(encoder::E; channels=3, nclasses=1, batch_norm=true) where {E}
    UNet(
        ConvBlock((3,3), channels, first(filters(encoder)), Flux.relu, batch_norm=batch_norm), 
        encoder, 
        nclasses=nclasses
    )
end
function UNet(input::I, encoder::E; nclasses=1, batch_norm=true) where {I,E}
    return UNet(
        input, 
        encoder, 
        build_decoder(filters(encoder), [64, 128, 256, 512], batch_norm), 
        Flux.Conv((1,1), 64=>nclasses)
    )
end

function (m::UNet)(x)
    return @pipe m.input(x) |> m.encoder |> reverse |> m.decoder |> last |> m.head
end

function build_decoder(encoder_filters, decoder_filters, batch_norm)
    Flux.PairwiseFusion(
        (a,b)->cat(a, b, dims=3), 
        Flux.ConvTranspose((2,2), encoder_filters[5]=>decoder_filters[4], stride=2), 
        decoder_block(decoder_filters[4], encoder_filters[4], decoder_filters[3], batch_norm), 
        decoder_block(decoder_filters[3], encoder_filters[3], decoder_filters[2], batch_norm), 
        decoder_block(decoder_filters[2], encoder_filters[2], decoder_filters[1], batch_norm), 
        ConvBlock((3,3), encoder_filters[1] + decoder_filters[1], decoder_filters[1], Flux.relu, batch_norm=batch_norm)
    )
end

function decoder_block(up_filters, skip_filters, out_filters, batch_norm)
    return Flux.Chain(
        ConvBlock((3,3), up_filters+skip_filters, up_filters, Flux.relu, batch_norm=batch_norm), 
        Flux.ConvTranspose((2,2), up_filters=>out_filters, stride=2), 
    )
end

"""
struct DecoderBlock{U,C}
    up::U
    conv::C
end

Flux.@layer DecoderBlock

function DecoderBlock(up_channels, skip_channels, filters; batch_norm=true)
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

function Decoder(encoder_filters, filters; batch_norm=true)
    Decoder(
        DecoderBlock(filters[2], encoder_filters[1], filters[1]; batch_norm=batch_norm), 
        DecoderBlock(filters[3], encoder_filters[2], filters[2]; batch_norm=batch_norm), 
        DecoderBlock(filters[4], encoder_filters[3], filters[3]; batch_norm=batch_norm), 
        DecoderBlock(encoder_filters[5], encoder_filters[4], filters[4]; batch_norm=batch_norm), 
    )
end

function(m::Decoder)(x1, x2, x3, x4, x5)
    decoder_4 = m.decoder4(x5, x4)
    decoder_3 = m.decoder3(decoder_4, x3)
    decoder_2 = m.decoder2(decoder_3, x2)
    decoder_1 = m.decoder1(decoder_2, x1)
    return decoder_1
end
"""