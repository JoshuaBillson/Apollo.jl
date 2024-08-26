"""
    SegmentationModel(;input=RasterInput(), encoder=StandardEncoder(), nclasses=1, batch_norm=true)

Construct a standard UNet model for semantic segmentation.

# Keywords
- `input`: The input block, which defaults to two convolutional layers as with standard UNet.
- `encoder`: The encoder to use for the UNet model. Defaults to the standard UNet encoder.
- `nclasses`: The number of output channels produced by the head. The model will use the
`sigmoid` activation function when `nclasses == 1` and `softmax` otherwise.
- `batch_norm`: Use batch normalization after each convolutional layer (default=true).
"""
struct SegmentationModel{I,E,D,H,F}
    input::I
    encoder::E
    decoder::D
    head::H
    activation::F
end

Flux.@layer SegmentationModel

function SegmentationModel(;input=RasterInput(), encoder=StandardEncoder(), nclasses=1, batch_norm=true)
    input_block = build_input(input, filters(encoder)[1])
    encoder_block = build_encoder(encoder)
    decoder_block = build_decoder(encoder, batch_norm)
    head_block = Flux.Conv((1,1), 64=>nclasses)
    activation = nclasses == 1 ? sigmoid : softmax
    return SegmentationModel(input_block, encoder_block, decoder_block, head_block, activation)
end

function (m::SegmentationModel)(x)
    input_out = m.input(x)
    encoder_out = Flux.activations(m.encoder, input_out) |> reverse
    decoder_out = m.decoder(encoder_out..., input_out) |> last
    return m.head(decoder_out) |> m.activation
end

features(m::SegmentationModel, x) = @pipe m.input(x) |> m.encoder |> reverse |> m.decoder |> last

function build_decoder(encoder::AbstractEncoder, batch_norm)
    enc_fs = filters(encoder)
    dec_fs = [64, 128, 256, 512, 1024, 2048, 4096][1:length(enc_fs)-1]
    Flux.PairwiseFusion(
        (a,b)->cat(a, b, dims=3), 
        Flux.ConvTranspose((2,2), enc_fs[end]=>dec_fs[end], stride=2), 
        [decoder_block(dec_fs[i], enc_fs[i], dec_fs[i-1], batch_norm) for i in length(enc_fs)-1:-1:2]..., 
        ConvBlock((3,3), enc_fs[1] + dec_fs[1], dec_fs[1], Flux.relu, batch_norm=batch_norm)
    )
end

function decoder_block(up_filters, skip_filters, out_filters, batch_norm)
    return Flux.Chain(
        ConvBlock((3,3), up_filters+skip_filters, up_filters, Flux.relu, batch_norm=batch_norm), 
        Flux.ConvTranspose((2,2), up_filters=>out_filters, stride=2), 
    )
end

sigmoid(x::AbstractArray{<:Real,4}) = Flux.sigmoid(x)

softmax(x::AbstractArray{<:Real,4}) = Flux.softmax(x, dims=3)