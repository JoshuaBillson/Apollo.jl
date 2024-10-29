function SwinTransformerBlock(dim, nheads; window_size=7, shift_size=0,
    mlp_ratio=4, qkv_bias=true, drop=0., attn_drop=0., drop_path=0.)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                WindowedAttention(dim, dim, window_size, nheads=nheads, qkv_bias=qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop)
            ), 
            +
        ), 
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim), 
                MLP(dim, dim*mlp_ratio, dim, drop)

            ), 
            +
        )
    )
end

function SWIN(;inchannels=3, depths=[2,2,6,2], embed_dim=96, nheads=[3,6,12,24], nclasses=1000, 
    window_size=7, mlp_ratio=4., qkv_bias=true, drop_rate=0.1, attn_drop_rate=0.1)

    dims = [embed_dim * 2^(i-1) for i in eachindex(depths)]
    Flux.Chain(
        Flux.Conv((4,4), inchannels=>embed_dim, stride=4, pad=Flux.SamePad()), 
        img2seq, 
        [SwinTransformerBlock(dims[1], nheads[1]) for _ in 1:depths[1]]...,
        PatchMerging(dims[1]),
        [SwinTransformerBlock(dims[2], nheads[2]) for _ in 1:depths[2]]..., 
        PatchMerging(dims[2]), 
        [SwinTransformerBlock(dims[3], nheads[3]) for _ in 1:depths[3]]..., 
        PatchMerging(dims[3]), 
        [SwinTransformerBlock(dims[4], nheads[4]) for _ in 1:depths[4]]..., 
        Flux.LayerNorm(dims[4]), 
        seq2img, 
        Flux.AdaptiveMeanPool((1,1)), 
        Flux.MLUtils.flatten, 
        Flux.Dense(dims[4] => nclasses)
    )
end