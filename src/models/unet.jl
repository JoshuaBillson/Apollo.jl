struct UNet{I,E,D,H}
    input::I
    encoder::E
    decoder::D
    head::H
end

Flux.@layer UNet

"""
    UNet(;encoder=StandardEncoder(batch_norm=true), input=nothing, channels=3, nclasses=1, batch_norm=true)

Construct a UNet model.

# Keywords
- `encoder`: The encoder to use for the UNet model. Defaults to the standard encoder.
- `input`: The input block, which defaults to two convolutional layers as with standard UNet.
- `channels`: The number of input channels to use when `input=nothing`. Ignored when `input` is specified.
- `nclasses`: The number of output channels produced by the head.
- `batch_norm`: Use batch normalization after each convolutional layer (default=true).
"""
function UNet(;encoder=StandardEncoder(batch_norm=true), input=nothing, channels=3, nclasses=1, batch_norm=true)
    input = isnothing(input) ? ConvBlock((3,3), channels, first(filters(encoder)), Flux.relu, batch_norm=batch_norm) : input
    UNet(
        input,
        encoder,
        nclasses=nclasses, 
        batch_norm=batch_norm
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

input(x::UNet) = x.input
encoder(x::UNet) = x.encoder
decoder(x::UNet) = x.decoder
head(x::UNet) = x.head

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