"""
    UNet(;encoder=ResNet(depth=34, weights=:ImageNet), inchannels=3, nclasses=1, batch_norm=true)

Construct a standard UNet model for semantic segmentation.

# Keywords
- `encoder`: The encoder to use for the UNet model. Defaults to ResNet34 with ImageNet weights.
- `inchannels`: The number of channels in the input tensor (default=3).
- `nclasses`: The number of output channels produced by the head.
- `batch_norm`: Use batch normalization after each convolutional layer (default=true).
"""
struct UNet{E,D,H}
    encoder::E
    decoder::D
    head::H
end

Flux.@layer :expand UNet

function UNet(;encoder=ResNet(depth=34, weights=:ImageNet), inchannels=3, nclasses=1, batch_norm=true, depth=5)
    _filters = filters(encoder)
    encoder = build_encoder(encoder, inchannels)[1:depth]
    decoder = _build_unet_decoder(_filters, batch_norm, depth)
    head = _build_unet_head(32, nclasses, true)
    return UNet(encoder, decoder, head)
end

function (m::UNet)(x)
    encoder_out = _unet_encoder_forward(m.encoder, x)
    decoder_out = _unet_decoder_forward(m.decoder, encoder_out[2:end], encoder_out[1])
    return m.head(decoder_out)
end

function _build_unet_decoder(encoder_filters, batch_norm, depth)
    encoder_filters = reverse(encoder_filters[1:depth])
    skip_filters = encoder_filters[2:end]
    decoder_filters = reverse((32, 64, 128, 256, 512)[1:depth-1])
    up_filters = (encoder_filters[1], decoder_filters[1:end-1]...)
    filters = zip(up_filters, skip_filters, decoder_filters) |> collect
    ntuple(length(filters)) do i
        up_filters = filters[i][1]
        skip_filters = filters[i][2]
        decoder_filters = filters[i][3]
        _unet_decoder_block(up_filters, skip_filters, decoder_filters, batch_norm)
    end
end

function _build_unet_head(infilters, nclasses, batch_norm)
    Flux.Chain(
        x -> Flux.upsample_nearest(x, (2,2)),
        ConvBlock(3, infilters, infilters, Flux.relu, batch_norm=batch_norm), 
        Flux.Conv((1,1), infilters=>nclasses)
    )
end

function _unet_decoder_block(up_filters, skip_filters, out_filters, batch_norm)
    Flux.Chain(
        Flux.PairwiseFusion(
            (a,b)->cat(a, b, dims=3), 
            x -> Flux.upsample_nearest(x, (2,2)),
            ConvBlock(3, up_filters+skip_filters, out_filters, Flux.relu, batch_norm=batch_norm)
        ), 
        last
    )
end

function _unet_encoder_forward(layers::Tuple, x)
    res = first(layers)(x)
    @debug size(res)
    return (_unet_encoder_forward(Base.tail(layers), res)..., res)
  end
 _unet_encoder_forward(::Tuple{}, x) = ()

 function _unet_decoder_forward(layers::Tuple, skip, x)
    res = first(layers)((x, first(skip)))
    @debug size(res)
    return _unet_decoder_forward(Base.tail(layers), Base.tail(skip), res)
 end
 _unet_decoder_forward(::Tuple{}, skip, x) = x
