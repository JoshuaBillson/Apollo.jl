"""
    SeparableConv(filter, in, out, σ=Flux.relu)

A separable convolutional layer consists of a depthwise convolution followed by a 1x1 convolution.

# Parameters
- `filter`: The size of the filter as a tuple.
- `in`: The number of input features.
- `out`: The number of output features.
- `σ`: The activation function to apply following the 1x1 convolution.
"""
function SeparableConv(kernel_size::Int, in::Int, out::Int, σ=Flux.relu)
    Flux.Chain(
        Flux.DepthwiseConv((kernel_size, kernel_size), in=>in, pad=Flux.SamePad()), 
        Flux.Conv((1,1), in=>out, σ, pad=Flux.SamePad())
    )
end

"""
    ConvBlock(filter, in, out, σ; depth=2, batch_norm=true)

A block of convolutional layers with optional batch normalization.

# Parameters
- `filter`: The size of the filter to use for each layer.
- `in`: The number of input features.
- `out`: The number of output features.
- `σ`: The activation function to apply after each convolution.
- `depth`: The number of successive convolutions in the block.
- `batch_norm`: Applies batch normalization after each convolution if `true`.
"""
function ConvBlock(kernel_size::Int, in::Int, out::Int, σ; depth=2, batch_norm=true)
    return Flux.Chain(
        [ifelse(
            i == 1, 
            Conv(kernel_size, in, out, σ, batch_norm=batch_norm), 
            Conv(kernel_size, out, out, σ, batch_norm=batch_norm)
            ) 
        for i in 1:depth]...
    )
end

function Conv(kernel_size::Int, in::Int, out::Int, σ; batch_norm=true, groups=1, dilation=1)
    if batch_norm
        return Flux.Chain(
            Flux.Conv((kernel_size, kernel_size), in=>out, pad=Flux.SamePad(), groups=groups, dilation=dilation),
            Flux.BatchNorm(out, σ), 
        )
    else
        return Flux.Conv((kernel_size, kernel_size), in=>out, σ, pad=Flux.SamePad())
    end
end

"""
    LSTM(in => out)

A stateless pixel-wise LSTM layer that expects an input tensor with shape WHCLN, where
W is image width, H is image height, C is image channels, L is sequence length, and N is 
batch size. The LSTM module will be applied to each pixel in the sequence from first to last 
with all but the final output discarded.

# Parameters
- `in`: The number of input features.
- `out`: The number of output features.

# Example
```jldoctest
julia> m = Apollo.LSTM(2=>32);

julia> x = rand(Float32, 64, 64, 2, 9, 4);

julia> m(x) |> size
(64, 64, 32, 4)
```
"""
struct LSTM{T}
    lstm::T
end

Flux.@layer LSTM

function LSTM(in_out::Pair{Int,Int})
    return LSTM(Flux.LSTM(in_out[1] => in_out[2]))
end

function (m::LSTM)(x::AbstractArray{<:AbstractFloat,5})
    # Reset LSTM State
    Flux.reset!(m.lstm)

    # Run Forward Pass
    @pipe permutedims(x, (3, 1, 2, 4, 5)) |>             # Permute to (CWHLN)
    [_[:,:,:,i,:] for i in 1:size(x, 4)] |>              # Split Time Series
    [reshape(x, (2,:)) for x in _] |>                    # Reshape to (CN)
    [m.lstm(x) for x in _] |>                            # Apply LSTM to each time stamp
    last |>                                              # Keep Last Prediction
    reshape(_, (:, size(x,1), size(x,2), size(x,5))) |>  # Reshape to (CWHN)
    permutedims(_, (2, 3, 1, 4))                         # Permute to (WHCN)
end

"""
    MultiHeadSelfAttention(planes::Integer, nheads::Integer = 8; qkv_bias::Bool = false, 
                attn_dropout_prob = 0., proj_dropout_prob = 0.)

Multi-head self-attention layer.

# Arguments

  - `planes`: number of input channels
  - `nheads`: number of heads
  - `qkv_bias`: whether to use bias in the layer to get the query, key and value
  - `attn_dropout_prob`: dropout probability after the self-attention layer
  - `proj_dropout_prob`: dropout probability after the projection layer
"""
struct MultiHeadSelfAttention{P, Q, R}
    nheads::Int
    qkv_layer::P
    attn_drop::Q
    projection::R
end
Flux.@layer :expand MultiHeadSelfAttention

function MultiHeadSelfAttention(inplanes::Int, outplanes::Int; nheads::Int = 8, qkv_bias::Bool = false, attn_dropout_prob = 0.0, proj_dropout_prob = 0.0)
    @assert outplanes % nheads==0 "planes should be divisible by nheads"
    qkv_layer = Flux.Dense(inplanes, outplanes * 3; bias = qkv_bias)
    attn_drop = Flux.Dropout(attn_dropout_prob)
    proj = Flux.Chain(Flux.Dense(outplanes, outplanes), Flux.Dropout(proj_dropout_prob))
    return MultiHeadSelfAttention(nheads, qkv_layer, attn_drop, proj)
end

function (m::MultiHeadSelfAttention)(x::AbstractArray{<:Number, 3})
    qkv = m.qkv_layer(x)
    q, k, v = Flux.chunk(qkv, 3, dims = 1)
    y, α = Flux.NNlib.dot_product_attention(q, k, v; m.nheads, fdrop = m.attn_drop)
    y = m.projection(y)
    return y
end

struct WindowedAttention{A}
    window_size::Integer
    att::A
end
Flux.@layer :expand WindowedAttention

function WindowedAttention(inplanes::Int, outplanes::Int, window_size::Int; nheads::Int = 8, qkv_bias::Bool = false, attn_dropout_prob = 0.0, proj_dropout_prob = 0.0)
    @assert outplanes % nheads==0 "planes should be divisible by nheads"
    return WindowedAttention(
        window_size, 
        MultiHeadSelfAttention(
            inplanes, outplanes; nheads=nheads, qkv_bias=qkv_bias, attn_dropout_prob=attn_dropout_prob, proj_dropout_prob=proj_dropout_prob
        )
    )
end

function (m::WindowedAttention)(x::AbstractArray{<:Number, 3})
    C, L, B = size(x)
    @pipe window_partition(x, m.window_size) |> img2seq |> m.att |> seq2img |> window_reverse(_, m.window_size, W, H)
end
function (m::WindowedAttention)(x::AbstractArray{<:Number, 4})
    W, H, C, B = size(x)
    @pipe window_partition(x, m.window_size) |> img2seq |> m.att |> seq2img |> window_reverse(_, m.window_size, W, H)
end

struct WindowTransformerBlock{A,M,C,N}
    att::A
    mlp::M
    conv::C
    norm1::N
    norm2::N
    norm3::N
    window_size::Integer
end

Flux.@layer :expand WindowTransformerBlock

function WindowTransformerBlock(dim, nheads; window_size=7, mlp_ratio=4, qkv_bias=true, drop=0.0, attn_drop=0.0)
    WindowTransformerBlock(
        MultiHeadSelfAttention(dim, dim; nheads=nheads, qkv_bias=qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop),
        MLP(dim, dim*mlp_ratio, dim, drop), 
        Flux.Conv((7,7), dim => dim, groups=dim, pad=Flux.SamePad()), 
        Flux.LayerNorm(dim), 
        Flux.LayerNorm(dim), 
        Flux.LayerNorm(dim), 
        window_size
    )
end

function (m::WindowTransformerBlock)(x)
    W, H, C, N = size(x)

    residual = @pipe permutedims(x, (3,1,2,4)) |> reshape(_, (C,W*H,N))
    x = @pipe residual |> 
    m.norm1 |> 
    reshape(_, (C,W,H,N)) |>
    window_partition(_, m.window_size) |> 
    reshape(_, (C,m.window_size*m.window_size,:)) |> 
    m.att |>
    reshape(_, (C,m.window_size,m.window_size,:)) |>
    window_reverse(_, m.window_size, W, H) |>
    reshape(_, (C,W*H,:))

    x = m.norm2(x .+ residual)

    residual = x
    x = @pipe reshape(x, (C,W,H,N)) |> permutedims(_, (2,3,1,4)) |> m.conv |> permutedims(_, (3,1,2,4)) |> reshape(_, (C,W*H,N))
    x = x .+ residual

    x = x .+ m.mlp(m.norm3(x))

    return @pipe reshape(x, (C,W,H,:)) |> permutedims(_, (2,3,1,4))
end

function WinTransformer(;depths=[2,2,6,2], embed_dim=96, nheads=[3,6,12,24], nclasses=1000)
    dims = [embed_dim * 2^(i-1) for i in eachindex(depths)]
    Flux.Chain(
        Flux.Conv((4,4), 3=>embed_dim, stride=4, pad=Flux.SamePad()), 
        [Apollo.WindowTransformerBlock(dims[1], nheads[1]) for _ in 1:depths[1]]...,
        Apollo.PatchMerging(dims[1]),
        [Apollo.WindowTransformerBlock(dims[2], nheads[2]) for _ in 1:depths[2]]..., 
        Apollo.PatchMerging(dims[2]), 
        [Apollo.WindowTransformerBlock(dims[3], nheads[3]) for _ in 1:depths[3]]..., 
        Apollo.PatchMerging(dims[3]), 
        [Apollo.WindowTransformerBlock(dims[4], nheads[4]) for _ in 1:depths[4]]..., 
        Flux.AdaptiveMeanPool((1,1)), 
        Flux.MLUtils.flatten, 
        Flux.Dense(dims[4] => nclasses)
    )
end

function MLP(indims, hiddendims, outdims, dropout)
    return Flux.Chain(
        Flux.Dense(indims => hiddendims, Flux.gelu), 
        Flux.Dropout(dropout), 
        Flux.Dense(hiddendims => outdims), 
        Flux.Dropout(dropout)
    )
end

function PatchMerging(dims::Int)
    Flux.Chain(
        merge_patches, 
        x -> permutedims(x, (3, 1, 2, 4)), 
        Flux.LayerNorm(4*dims), 
        Flux.Dense(dims*4=>dims*2, bias=false), 
        x -> permutedims(x, (2, 3, 1, 4)), 
    )
end

img2seq(x) = permutedims(reshape(x, (:, size(x, 3), size(x, 4))), (2, 1, 3))

function seq2img(x)
    s = round(Int, sqrt(size(x, 2)))
    @pipe permutedims(x, (2, 1, 3)) |> reshape(_, (s, s, size(x, 1), size(x, 3)))
end

function window_partition(x, window_size)
    C, W, H, B = size(x)
    @pipe reshape(x, (C, window_size, W ÷ window_size, window_size, H ÷ window_size, B)) |> 
    permutedims(_, (1,2,4,3,5,6)) |> 
    reshape(_, (C, window_size, window_size, :))
end

function window_reverse(x, window_size, W, H)
    C = size(x, 1)
    @pipe reshape(x, (C, window_size, window_size, W ÷ window_size, H ÷ window_size, :)) |>
    permutedims(_, (1,2,4,3,5,6)) |>
    reshape(_, (C,W,H,:))
end

function merge_patches(x::AbstractArray{<:Number,4})
    x1 = x[1:2:end,1:2:end,:,:]
    x2 = x[2:2:end,1:2:end,:,:]
    x3 = x[1:2:end,2:2:end,:,:]
    x4 = x[2:2:end,2:2:end,:,:]
    return cat(x1, x2, x3, x4, dims=3)
end