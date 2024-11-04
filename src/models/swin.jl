function SwinTransformerBlock(dim, input_resolution, nheads; window_size=7, window_shift=0,
    mlp_ratio=4, qkv_bias=true, drop=0., attn_drop=0., drop_path=0.)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                WindowedAttention(dim, dim, input_resolution, window_size; nheads, window_shift, qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop)
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

function SwinLayer(embed_dim, input_resolution, depth, nheads, window_size, shift, mlp_ratio, qkv_bias, drop, attn_drop, patch_merging)
    window_shift = shift ? window_size ÷ 2 : 0
    blocks = [
        SwinTransformerBlock(
            embed_dim, 
            input_resolution,
            nheads; 
            window_size, 
            mlp_ratio, 
            qkv_bias, 
            drop, 
            attn_drop, 
            window_shift = ((i % 2) == 0) ? window_shift : 0
            )
        for i in 1:depth]
    return patch_merging ? Flux.Chain(blocks..., PatchMerging(embed_dim)) : Flux.Chain(blocks...)
end

function SWIN(;inchannels=3, depths=[2,2,6,2], embed_dim=96, nheads=[3,6,12,24], nclasses=1000, 
    windows=[7,7,7,7], mlp_ratio=4, qkv_bias=true, drop_rate=0.1, attn_drop_rate=0.1, shift_windows=true)

    nlayers = length(depths)
    dims = [embed_dim * 2^(i-1) for i in eachindex(depths)]
    resolutions = [(224 ÷ 4) ÷ (2 ^ (i-1)) for i in eachindex(depths)]
    Flux.Chain(
        Flux.Conv((4,4), inchannels=>embed_dim, stride=4, pad=Flux.SamePad()), 
        img2seq, 
        [SwinLayer(dims[i], resolutions[i], depths[i], nheads[i], windows[i], shift_windows, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, i < nlayers) for i in 1:nlayers]..., 
        Flux.LayerNorm(dims[4]), 
        seq2img, 
        Flux.AdaptiveMeanPool((1,1)), 
        Flux.MLUtils.flatten, 
        Flux.Dense(dims[4] => nclasses)
    )
end