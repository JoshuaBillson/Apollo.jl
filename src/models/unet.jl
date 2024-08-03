"""
    UNet(;encoder=StandardEncoder(batch_norm=true), input=nothing, channels=3, nclasses=1, batch_norm=true, activation=identity)

Construct a UNet model.

# Keywords
- `encoder`: The encoder to use for the UNet model. Defaults to the standard encoder.
- `input`: The input block, which defaults to two convolutional layers as with standard UNet.
- `channels`: The number of input channels to use when `input=nothing`. Ignored when `input` is specified.
- `nclasses`: The number of output channels produced by the head.
- `batch_norm`: Use batch normalization after each convolutional layer (default=true).
"""
struct UNet{I,E,D,H,F}
    input::I
    encoder::E
    decoder::D
    head::H
    activation::F
end

Flux.@layer UNet

function UNet(;encoder=StandardEncoder(batch_norm=true), input=nothing, channels=3, nclasses=1, batch_norm=true, activation=identity)
    input = isnothing(input) ? ConvBlock((3,3), channels, first(filters(encoder)), Flux.relu, batch_norm=batch_norm) : input
    UNet(
        input,
        encoder,
        activation, 
        nclasses, 
        batch_norm
    )
end
function UNet(input::I, encoder::E, activation::F, nclasses::Int, batch_norm::Bool) where {I,E,F}
    return UNet(
        input, 
        encoder, 
        build_decoder(filters(encoder), [64, 128, 256, 512], batch_norm), 
        Flux.Conv((1,1), 64=>nclasses), 
        activation
    )
end

function (m::UNet)(x)
    return @pipe m.input(x) |> m.encoder |> reverse |> m.decoder |> last |> m.head |> m.activation
end

features(m::UNet, x) = @pipe m.input(x) |> m.encoder |> reverse |> m.decoder |> last

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