function PRM(kernel_size, in_features, embedding, stride, dilations)
    kernel = (kernel_size, kernel_size)
    filters = in_features => embedding
    Flux.Parallel(
        (x...) -> cat(x..., dims=3), 
        [Flux.Conv(kernel, filters, Flux.gelu, stride=stride, pad=Flux.SamePad(), dilation=d) for d in dilations]...
    )
end

function PCM(in_dims, hidden_dims, out_dims, downsample, groups)
    strides = [(downsample - 2 * i) >= 0 ? 2 : 1 for i in 1:3]
    Flux.Chain(
        Flux.Conv((3,3), in_dims => hidden_dims, pad=Flux.SamePad(), groups=groups, stride=strides[1]), 
        Flux.BatchNorm(hidden_dims, Flux.swish),
        Flux.Conv((3,3), hidden_dims => hidden_dims, pad=Flux.SamePad(), groups=groups, stride=strides[2]), 
        Flux.BatchNorm(hidden_dims, Flux.swish), 
        Flux.Conv((3,3), hidden_dims => out_dims, pad=Flux.SamePad(), groups=groups, stride=strides[3]), 
    )
end

function NormalPCM(in_dims, hidden_dims, out_dims, downsample, groups)
    strides = [(downsample - 2 * i) >= 0 ? 2 : 1 for i in 1:3]
    Flux.Chain(
        Flux.Conv((3,3), in_dims => hidden_dims, pad=Flux.SamePad(), groups=groups, stride=strides[1]), 
        Flux.BatchNorm(hidden_dims, Flux.swish),
        Flux.Conv((3,3), hidden_dims => out_dims, pad=Flux.SamePad(), groups=groups, stride=strides[2]), 
        Flux.BatchNorm(out_dims, Flux.swish), 
        Flux.Conv((3,3), out_dims => out_dims, pad=Flux.SamePad(), groups=groups, stride=strides[3]), 
    )
end

function AttentionLayer(in_planes, out_planes; nheads=1, qkv_bias=false, attn_dropout_prob=0.0, proj_dropout_prob=0.0, windows=false)
    if windows
        WindowedAttention(in_planes, out_planes, 7; nheads=nheads, qkv_bias=qkv_bias, attn_dropout_prob=attn_dropout_prob, proj_dropout_prob=proj_dropout_prob)
    else
        MultiHeadSelfAttention(in_planes, out_planes; nheads=nheads, qkv_bias=qkv_bias, attn_dropout_prob=attn_dropout_prob, proj_dropout_prob=proj_dropout_prob)
    end
end

function ReductionBlock(in_dims, embed_dims, token_dims; nheads=1, kernel_size=3, downsample=2, dilations=[1,2,3,4], groups=1, mlp_ratio=1, qkv_bias=false, drop=0.0, attn_drop=0.0, windows=false)
    Flux.Chain(
        Flux.Parallel(
            +, 
            Flux.Chain(PCM(in_dims, embed_dims, token_dims, downsample, groups), img2seq), 
            Flux.Chain(
                PRM(kernel_size, in_dims, embed_dims, downsample, dilations), 
                img2seq, 
                Flux.LayerNorm(embed_dims * length(dilations)), 
                AttentionLayer(embed_dims * length(dilations), token_dims; nheads=nheads, qkv_bias=qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop, windows=windows)
            )
        ), 
        Flux.SkipConnection(
            Flux.Chain(Flux.LayerNorm(token_dims), MLP(token_dims, token_dims * mlp_ratio, token_dims, drop)), 
            +
        ), 
        seq2img
    )
end

function NormalBlock(dims; nheads=1, groups=1, mlp_ratio=4, qkv_bias=false, drop=0.0, attn_drop=0.0, windows=false)
    Flux.Chain(
        Flux.Parallel(
            +, 
            Flux.Chain(seq2img, NormalPCM(dims, dims * mlp_ratio, dims, 1, groups), img2seq), 
            Flux.Chain(
                Flux.LayerNorm(dims), 
                AttentionLayer(dims, dims; nheads=nheads, qkv_bias=qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop, windows=windows)
            )
        ), 
        Flux.SkipConnection(
            Flux.Chain(Flux.LayerNorm(dims), MLP(dims, dims * mlp_ratio, dims, drop)), 
            +
        )
    )
end

function VitaeV2(;nclasses=1000)
    Flux.Chain(
        ReductionBlock(3, 64, 64, nheads=1, kernel_size=7, downsample=4, groups=1, windows=true), 
        img2seq, 
        [NormalBlock(64, nheads=1, groups=1, windows=true) for _ in 1:2]..., 
        seq2img, 
        ReductionBlock(64, 64, 128, nheads=1, kernel_size=3, downsample=2, dilations=[1,2,3], groups=16, windows=true), 
        img2seq, 
        [NormalBlock(128, nheads=2, groups=32, windows=true) for _ in 1:2]..., 
        seq2img, 
        ReductionBlock(128, 128, 256, nheads=2, kernel_size=3, downsample=2, dilations=[1,2], groups=32), 
        img2seq, 
        [NormalBlock(256, nheads=4, groups=64) for _ in 1:8]..., 
        seq2img, 
        ReductionBlock(256, 256, 512, nheads=4, kernel_size=3, downsample=2, dilations=[1,2], groups=64), 
        img2seq, 
        [NormalBlock(512, nheads=8, groups=128) for _ in 1:2]..., 
        x -> dropdims(mean(x, dims=2), dims=2), 
        Flux.Dense(512=>nclasses)
    )
end