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
    MultiHeadSelfAttention(inplanes::Int, outplanes::Int; nheads::Int = 8, qkv_bias::Bool = false, attn_dropout_prob = 0.0, proj_dropout_prob = 0.0)

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
    #C, L, N = size(x)
    #qkv = reshape(m.qkv_layer(x), (:, 3, L, N))
    #q = reshape(qkv[:,1:1,:,:], (:,L,N))
    #k = reshape(qkv[:,2:2,:,:], (:,L,N))
    #v = reshape(qkv[:,3:3,:,:], (:,L,N))
    q, k, v = Flux.chunk(qkv, 3, dims = 1)
    y, α = Flux.NNlib.dot_product_attention(q, k, v; m.nheads, fdrop = m.attn_drop)
    y = m.projection(y)
    return y
end

struct WindowedAttention{P,R,D,M,B,I}
    nheads::Int
    window_size::Int
    window_shift::Int
    relative_position_index::I
    relative_position_bias::B
    attention_mask::M
    qkv_layer::P
    attn_drop::D
    projection::R
end

Flux.@layer :expand WindowedAttention trainable=(relative_position_bias, qkv_layer, attn_drop, projection)

function WindowedAttention(inplanes::Int, outplanes::Int, feature_size::Int, window_size::Int; nheads::Int = 8, qkv_bias::Bool = false, attn_dropout_prob = 0.0, proj_dropout_prob = 0.0, window_shift=0)
    @assert outplanes % nheads==0 "planes should be divisible by nheads"
    @assert window_shift >= 0 "window shift must be an integer greater than or equal to zero"

    # Initialize Layers
    qkv_layer = Flux.Dense(inplanes, outplanes * 3; bias = qkv_bias)
    attn_drop = Flux.Dropout(attn_dropout_prob)
    proj = Flux.Chain(Flux.Dense(outplanes, outplanes), Flux.Dropout(proj_dropout_prob))

    # Initialize Positional Embedding
    relative_position_bias = zeros(Float32, nheads, (2*window_size-1)*(2*window_size-1))

    # Compute Relative Position Indices
    relative_position_index = compute_relative_position_index((window_size,window_size))

    # Compute Attention Mask for Shifted Windows
    attention_mask = window_shift > 0 ? compute_attention_mask(window_size, (feature_size, feature_size)) : nothing

    # Construct Layer
    return WindowedAttention(
        nheads,
        window_size, 
        window_shift,
        relative_position_index, 
        relative_position_bias,
        attention_mask,
        qkv_layer, 
        attn_drop, 
        proj
    )
end

function (m::WindowedAttention)(x::AbstractArray{<:Number, 3})
    # Get Tensor Dimensions
    C, L, N = size(x)
    W = round(Int, sqrt(L))

    # Partition Into Windows (CxW*WxN)
    windows = if m.window_shift > 0
        shifted = circshift(reshape(x, (C,W,W,N)), (0,-m.window_shift,-m.window_shift,0))
        window_partition(reshape(shifted, (C,L,N)), m.window_size)
    else
        window_partition(x, m.window_size)
    end

    # Get Relative Position Bias
    relative_position_index = reshape(m.relative_position_index, :)
    relative_position_bias = m.relative_position_bias[:,relative_position_index]
    relative_position_bias = reshape(relative_position_bias, (:, m.window_size^2, m.window_size^2)) # [nH x Ww*Wh x Ww*Wh]
    relative_position_bias = permutedims(relative_position_bias, (2,3,1)) # [Ww*Wh x Ww*Wh x nH]
    relative_position_bias = Flux.unsqueeze(relative_position_bias, dims=4) # [Ww*Wh x Ww*Wh x nH x 1]

    # Get Attention Mask
    attn_mask = nothing
    if !isnothing(m.attention_mask)
        nW = size(m.attention_mask, 3)
        wL = size(m.attention_mask, 1)
        attn_mask = Flux.zeros_like(m.attention_mask, Bool, (wL,wL,m.nheads,nW,N)) .| reshape(m.attention_mask, (wL,wL,1,nW,1))
        attn_mask = reshape(attn_mask, (wL,wL,m.nheads,:))
    end

    # Compute Attention
    qkv = m.qkv_layer(windows)
    q, k, v = Flux.chunk(qkv, 3, dims = 1)
    y, α = Flux.NNlib.dot_product_attention(q, k, v, relative_position_bias; nheads=m.nheads, fdrop=m.attn_drop, mask=attn_mask)
    y = m.projection(y)

    # Reverse Windows
    C = size(y, 1)
    if m.window_shift > 0
        shifted = circshift(reshape(y, (C,W,W,N)), (0,m.window_shift,m.window_shift,0))
        return @pipe window_reverse(shifted, m.window_size, W, W) |> reshape(_, (C,:,N))
    else
        return @pipe window_reverse(y, m.window_size, W, W) |> reshape(_, (C,:,N))
    end
end

"""
struct WindowTransformerBlock{A,M,C,N}
    att::A
    mlp::M
    conv::C
    norm1::N
    norm2::N
    norm3::N
    window_size::Int
end

Flux.@layer :expand WindowTransformerBlock

function WindowTransformerBlock(dim, nheads; window_size=7, mlp_ratio=4, qkv_bias=true, drop=0.0, attn_drop=0.0)
    WindowTransformerBlock(
        WindowedAttention(dim, dim, 7; nheads=nheads, qkv_bias=qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop),
        MLP(dim, dim*mlp_ratio, dim, drop), 
        Flux.Conv((7,7), dim => dim, groups=dim, pad=Flux.SamePad()), 
        Flux.LayerNorm(dim), 
        Flux.LayerNorm(dim), 
        Flux.LayerNorm(dim), 
        window_size
    )
end
"""

function WinTransformerBlock(dim, nheads; window_size=7, mlp_ratio=4, qkv_bias=true, drop=0.1, attn_drop=0.1)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim), 
                WindowedAttention(dim, dim, window_size; nheads=nheads, qkv_bias=qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop),
            ), 
            +
        ), 
        Flux.LayerNorm(dim), 
        Flux.SkipConnection(
            Flux.Chain(
                seq2img, 
                Flux.Conv((3,3), dim => dim, groups=1, dilation=3, pad=Flux.SamePad()), 
                img2seq
            ), 
            +
        ), 
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim), 
                MLP(dim, dim*mlp_ratio, dim, drop), 
            ), 
            +
        )
    )
end

"""
function (m::WindowTransformerBlock)(x)
    # First Block
    residual = x
    x = residual .+ m.att(m.norm1(x))

    # Second Block
    residual = m.norm2(x)
    x = @pipe seq2img(x) |> m.conv |> img2seq
    x = residual .+ x

    # Third Block
    residual = x
    x = m.norm3(x) |> m.mlp
    return residual .+ x
end
"""
function WinTransformer(;depths=[2,2,6,2], embed_dim=96, nheads=[3,6,12,24], nclasses=1000)
    dims = [embed_dim * 2^(i-1) for i in eachindex(depths)]
    Flux.Chain(
        Flux.Conv((4,4), 3=>embed_dim, stride=4, pad=Flux.SamePad()), 
        Flux.Dropout(0.1), 
        img2seq, 
        [Apollo.WinTransformerBlock(dims[1], nheads[1]) for _ in 1:depths[1]]...,
        Apollo.PatchMerging(dims[1]),
        [Apollo.WinTransformerBlock(dims[2], nheads[2]) for _ in 1:depths[2]]..., 
        Apollo.PatchMerging(dims[2]), 
        [Apollo.WinTransformerBlock(dims[3], nheads[3]) for _ in 1:depths[3]]..., 
        Apollo.PatchMerging(dims[3]), 
        [Apollo.WinTransformerBlock(dims[4], nheads[4]) for _ in 1:depths[4]]..., 
        seq2img, 
        Flux.AdaptiveMeanPool((1,1)), 
        Flux.MLUtils.flatten, 
        Flux.LayerNorm(dims[4]),
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
        Flux.LayerNorm(4*dims), 
        Flux.Dense(dims*4=>dims*2, bias=false), 
    )
end

img2seq(x) = permutedims(reshape(x, (:, size(x, 3), size(x, 4))), (2, 1, 3))

function seq2img(x)
    s = round(Int, sqrt(size(x, 2)))
    @pipe permutedims(x, (2, 1, 3)) |> reshape(_, (s, s, size(x, 1), size(x, 3)))
end

function window_partition(x::AbstractArray{<:Number,3}, window_size)
    C, L, B = size(x)
    S = round(Int, sqrt(L))
    @pipe reshape(x, (C,S,S,B)) |> window_partition(_, window_size) |> reshape(_, (C,window_size^2,:))
end
function window_partition(x::AbstractArray{<:Number,4}, window_size)
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

function merge_patches(x::AbstractArray{<:Number,3})
    C, L, N = size(x)
    S = round(Int, sqrt(L))
    @pipe reshape(x, (C,S,S,N)) |> merge_patches |> reshape(_, (C*4,:,N))
end
function merge_patches(x::AbstractArray{<:Number,4})
    x1 = @view x[:,1:2:end,1:2:end,:]
    x2 = @view x[:,2:2:end,1:2:end,:]
    x3 = @view x[:,1:2:end,2:2:end,:]
    x4 = @view x[:,2:2:end,2:2:end,:]
    return cat(x1, x2, x3, x4, dims=1)
end

function compute_relative_position_index(window_size)
    # Generate coordinates
    Wh, Ww = window_size
    coords_h = collect(0:Wh-1)
    coords_w = collect(0:Ww-1)
    coords = hcat(map(collect, Iterators.product(coords_h, coords_w))...)  # Shape: 2, Wh*Ww

    # Flatten coordinates
    coords_flatten = reshape(reverse(coords, dims=1), 2, Wh * Ww)

    # Compute relative coordinates
    relative_coords = Flux.unsqueeze(coords_flatten, 3) .- Flux.unsqueeze(coords_flatten, 2)
    relative_coords = permutedims(relative_coords, (2,3,1))
    relative_coords[:, :, 1] .+= Wh - 1  # shift to start from 0
    relative_coords[:, :, 2] .+= Ww - 1
    relative_coords[:, :, 1] .*= 2 * Ww - 1

    # Calculate relative position index
    relative_position_index = dropdims(sum(relative_coords, dims=3), dims=3)
    relative_position_index = relative_position_index .+ 1  # adjust for 1-based indexing
    return permutedims(relative_position_index, (2,1))  # adjust for column-major ordering
end

function dot_product_attention_scores(q::AbstractArray{T,4}, k::AbstractArray{T,4}, bias=nothing;
    fdrop=identity, mask=nothing) where T

    # The following permutedims and batched_mul are equivalent to
    # @tullio logits[j, i, h, b] := q[d, h, i, b] * k[d, h, j, b] / √T(qk_dim)
    kt = permutedims(k, (3, 1, 2, 4))
    qt = permutedims(q, (1, 3, 2, 4)) ./ √T(size(q, 1))
    logits = Flux.batched_mul(kt, qt) # [logits] = [kv_len, q_len, nheads, batch_size]

    logits = Flux.NNlib.apply_attn_bias(logits, bias)
    logits = Flux.NNlib.apply_attn_mask(logits, mask)

    α = Flux.softmax(logits, dims=1)
    return fdrop(α)
end

function dot_product_attention(q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4}, bias, fdrop, mask) where T
    # [q] = [qk_dim ÷ nheads, nheads, q_len, batch_size]
    # [k] = [qk_dim ÷ nheads, nheads, kv_len, batch_size]
    # [v] = [v_dim ÷ nheads, nheads, kv_len, batch_size]

    α = dot_product_attention_scores(q, k, bias; fdrop, mask)
    # [α] = [kv_len, q_len, nheads, batch_size]

    # The following permutedims and batched_mul are equivalent to
    # @tullio x[d, h, i, b] := α[j, i, h, b] * v[d, h, j, b]
    vt = permutedims(v, (1, 3, 2, 4))
    x = Flux.NNlib.batched_mul(vt, α)
    x = permutedims(x, (1, 3, 2, 4))
    # [x] = [kv_dim ÷ nheads, nheads, q_len, batch_size]
    return x, α
end

function compute_attention_mask(window_size::Int, feature_size::Tuple{Int,Int})
    W, H = feature_size
    shift_size = window_size ÷ 2
    img_mask = zeros(Float32, 1, W, H, 1)
    w_slices = [1:(W-window_size), (W-window_size+1):(W-shift_size), (W-shift_size+1):W]
    h_slices = [1:(H-window_size), (H-window_size+1):(H-shift_size), (H-shift_size+1):H]
    cnt = 0
    for w in w_slices
        for h in h_slices
            img_mask[:, w, h, :] .= cnt
            cnt += 1
        end
    end

    mask_windows = window_partition(img_mask, window_size)  # 1, window_size, window_size, nW
    mask_windows = reshape(mask_windows, (window_size^2, :))  # window_size * window_size, nW
    attn_mask = Flux.unsqueeze(mask_windows, dims=2) .- Flux.unsqueeze(mask_windows, dims=1)
    return attn_mask .== 0
end