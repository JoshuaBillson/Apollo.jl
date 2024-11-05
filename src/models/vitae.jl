function PRM(kernel_size, in_features, embedding, stride, dilations)
    kernel = (kernel_size, kernel_size)
    filters = in_features => embedding
    Flux.Parallel(
        _cat_prm,
        [Flux.Conv(kernel, filters, Flux.gelu, stride=stride, pad=Flux.SamePad(), dilation=d) for d in dilations]...
    )
end

_cat_prm(x...) = cat(x..., dims=3)

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

function NormalPCM(in_dims, hidden_dims, out_dims, groups)
    Flux.Chain(
        Flux.Conv((3,3), in_dims => hidden_dims, pad=Flux.SamePad(), groups=groups), 
        Flux.BatchNorm(hidden_dims, Flux.swish),
        Flux.Conv((3,3), hidden_dims => out_dims, pad=Flux.SamePad(), groups=groups), 
        Flux.BatchNorm(out_dims, Flux.swish), 
        Flux.Conv((3,3), out_dims => out_dims, pad=Flux.SamePad(), groups=groups), 
    )
end

function AttentionLayer(in_planes, out_planes; nheads=1, qkv_bias=false, attn_dropout_prob=0.0, proj_dropout_prob=0.0, windows=false)
    if windows
        WindowedAttention(in_planes, out_planes, 224, 7; nheads, qkv_bias, attn_dropout_prob, proj_dropout_prob)
    else
        MultiHeadSelfAttention(in_planes, out_planes; nheads, qkv_bias, attn_dropout_prob, proj_dropout_prob)
    end
end

function ReductionBlock(in_dims, embed_dims, token_dims; nheads=1, kernel_size=3, downsample=2, dilations=[1,2,3,4], groups=1, mlp_ratio=1, qkv_bias=false, drop=0.0, attn_drop=0.0, windows=false)
    Flux.Chain(
        Flux.Parallel(
            +, 
            Flux.Chain(
                PRM(kernel_size, in_dims, embed_dims, downsample, dilations), 
                img2seq, 
                Flux.LayerNorm(embed_dims * length(dilations)), 
                AttentionLayer(embed_dims * length(dilations), token_dims; nheads, qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop, windows=windows)
            )
        ), 
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(token_dims), 
                MLP(token_dims, token_dims * mlp_ratio, token_dims, drop)
            ), 
            +
        ), 
        seq2img
    )
end

function ReductionBlockV2(in_dims, embed_dims, token_dims; nheads=1, kernel_size=3, downsample=2, dilations=[1,2,3,4], groups=1, mlp_ratio=1, qkv_bias=false, drop=0.0, attn_drop=0.0, windows=true)
    Flux.Chain(
        Flux.Chain(
            seq2img, 
            PRM(kernel_size, in_dims, embed_dims, downsample, dilations), 
            Flux.Conv((1,1), embed_dims*length(dilations) => token_dims), 
            img2seq
        ), 
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(token_dims), 
                AttentionLayer(token_dims, token_dims; nheads, qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop, windows=windows)
            ), 
            +
        ), 
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(token_dims), 
                MLP(token_dims, token_dims * mlp_ratio, token_dims, drop)
            ), 
            +
        ), 
    )
end

function NormalBlock(dims; nheads=1, groups=1, mlp_ratio=4, qkv_bias=false, drop=0.0, attn_drop=0.0, windows=false)
    Flux.Chain(
        Flux.Parallel(
            +, 
            Flux.Chain(seq2img, NormalPCM(dims, dims * mlp_ratio, dims, groups), img2seq), 
            Flux.Chain(
                Flux.LayerNorm(dims), 
                AttentionLayer(dims, dims; nheads, qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop, windows=windows)
            )
        ), 
        Flux.SkipConnection(
            Flux.Chain(Flux.LayerNorm(dims), MLP(dims, dims * mlp_ratio, dims, drop)), 
            +
        )
    )
end

function NormalBlockV2(dims; nheads=1, groups=1, mlp_ratio=4, qkv_bias=false, drop=0.0, attn_drop=0.0, windows=true)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dims), 
                AttentionLayer(dims, dims; nheads, qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop, windows=windows)
            ), 
            +
        ), 
        Flux.SkipConnection(
            Flux.Chain(Flux.LayerNorm(dims), MLP(dims, dims * mlp_ratio, dims, drop)), 
            +
        )
    )
end


function VitaeV2(;inchannels=3, depths=[2,2,8,2], reduction_heads=[1,1,2,4], normal_heads=[1,2,4,8],
    embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], normal_groups=[1, 32, 64, 128], windows=[true,true,false,false],
    reduction_groups=[1, 16, 32, 64], mlp_ratio=4, qkv_bias=true, drop=0.1, attn_drop=0.1, nclasses=1000)
    Flux.Chain(
        ReductionBlock(inchannels, embed_dims[1], token_dims[1]; nheads=reduction_heads[1], groups=reduction_groups[1], windows=windows[1], kernel_size=7, downsample=4, mlp_ratio, qkv_bias, attn_drop, drop), 
        img2seq, 
        [NormalBlock(token_dims[1]; nheads=normal_heads[1], groups=normal_groups[1], windows=windows[1], mlp_ratio, qkv_bias, drop, attn_drop) for _ in 1:depths[1]]..., 
        seq2img, 
        ReductionBlock(token_dims[1], embed_dims[2], token_dims[2]; nheads=reduction_heads[2], groups=reduction_groups[2], windows=windows[2], dilations=[1,2,3], kernel_size=3, downsample=2, mlp_ratio, qkv_bias, attn_drop, drop), 
        img2seq, 
        [NormalBlock(token_dims[2]; nheads=normal_heads[2], groups=normal_groups[2], windows=windows[2], mlp_ratio, qkv_bias, drop, attn_drop) for _ in 1:depths[2]]..., 
        seq2img, 
        ReductionBlock(token_dims[2], embed_dims[3], token_dims[3]; nheads=reduction_heads[3], groups=reduction_groups[3], windows=windows[3], dilations=[1,2], kernel_size=3, downsample=2, mlp_ratio, qkv_bias, attn_drop, drop), 
        img2seq, 
        [NormalBlock(token_dims[3]; nheads=normal_heads[3], groups=normal_groups[3], windows=windows[3], mlp_ratio, qkv_bias, drop, attn_drop) for _ in 1:depths[3]]..., 
        seq2img, 
        ReductionBlock(token_dims[3], embed_dims[4], token_dims[4]; nheads=reduction_heads[4], groups=reduction_groups[4], windows=windows[4], dilations=[1,2], kernel_size=3, downsample=2, mlp_ratio, qkv_bias, attn_drop, drop), 
        img2seq, 
        [NormalBlock(token_dims[4]; nheads=normal_heads[4], groups=normal_groups[4], windows=windows[4], mlp_ratio, qkv_bias, drop, attn_drop) for _ in 1:depths[4]]..., 
        x -> dropdims(mean(x, dims=2), dims=2), 
        Flux.Dense(token_dims[4]=>nclasses)
    )
end

function VitaeV3(;inchannels=3, depths=[2,2,8,2], reduction_heads=[1,1,2,4], normal_heads=[1,2,4,8],
    embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], normal_groups=[1, 32, 64, 128], windows=[true,true,true,true],
    reduction_groups=[1, 16, 32, 64], mlp_ratio=4, qkv_bias=true, drop=0.1, attn_drop=0.1, nclasses=1000)
    Flux.Chain(
        img2seq,
        ReductionBlockV2(inchannels, embed_dims[1], token_dims[1]; nheads=reduction_heads[1], groups=reduction_groups[1], kernel_size=7, downsample=4, mlp_ratio, qkv_bias, attn_drop, drop), 
        [NormalBlockV2(token_dims[1]; nheads=normal_heads[1], groups=normal_groups[1], mlp_ratio, qkv_bias, drop, attn_drop) for _ in 1:depths[1]]..., 
        ReductionBlockV2(token_dims[1], embed_dims[2], token_dims[2]; nheads=reduction_heads[2], groups=reduction_groups[2], windows=windows[2], dilations=[1,2,3], kernel_size=3, downsample=2, mlp_ratio, qkv_bias, attn_drop, drop), 
        [NormalBlockV2(token_dims[2]; nheads=normal_heads[2], groups=normal_groups[2], windows=windows[2], mlp_ratio, qkv_bias, drop, attn_drop) for _ in 1:depths[2]]..., 
        ReductionBlockV2(token_dims[2], embed_dims[3], token_dims[3]; nheads=reduction_heads[3], groups=reduction_groups[3], windows=windows[3], dilations=[1,2], kernel_size=3, downsample=2, mlp_ratio, qkv_bias, attn_drop, drop), 
        [NormalBlockV2(token_dims[3]; nheads=normal_heads[3], groups=normal_groups[3], windows=windows[3], mlp_ratio, qkv_bias, drop, attn_drop) for _ in 1:depths[3]]..., 
        ReductionBlockV2(token_dims[3], embed_dims[4], token_dims[4]; nheads=reduction_heads[4], groups=reduction_groups[4], windows=windows[4], dilations=[1,2], kernel_size=3, downsample=2, mlp_ratio, qkv_bias, attn_drop, drop), 
        [NormalBlockV2(token_dims[4]; nheads=normal_heads[4], groups=normal_groups[4], windows=windows[4], mlp_ratio, qkv_bias, drop, attn_drop) for _ in 1:depths[4]]..., 
        x -> dropdims(mean(x, dims=2), dims=2), 
        Flux.Dense(token_dims[4]=>nclasses)
    )
end