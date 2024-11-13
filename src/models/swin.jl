function SwinTransformerBlock(dim, nheads; imsize=224, window_size=7, window_shift=0,
    mlp_ratio=4, qkv_bias=true, drop=0., attn_drop=0., drop_path=0.)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                WindowedAttention(dim, dim; imsize, window_size, nheads, window_shift, qkv_bias, attn_dropout_prob=attn_drop, proj_dropout_prob=drop)
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

function SwinLayer(dim, imsize, depth, nheads, window_size, shift, mlp_ratio, qkv_bias, drop, attn_drop, patch_merging)
    window_shift = shift ? window_size ÷ 2 : 0
    blocks = [
        SwinTransformerBlock(
            dim, 
            nheads; 
            imsize,
            window_size, 
            mlp_ratio, 
            qkv_bias, 
            drop, 
            attn_drop, 
            window_shift = ((i % 2) == 0) ? window_shift : 0
            )
        for i in 1:depth]
    return patch_merging ? Flux.Chain(blocks..., PatchMerging(dim)) : Flux.Chain(blocks...)
end

function swin(embed_dims, depths, nheads; 
    inchannels=3, nclasses=1000, windows=[7,7,7,7], mlp_ratio=4, qkv_bias=true, drop_rate=0.1, attn_drop_rate=0.1, shift_windows=true)

    nlayers = length(depths)
    resolutions = [(224 ÷ 4) ÷ (2 ^ (i-1)) for i in eachindex(depths)]
    Flux.Chain(
        Flux.Conv((4,4), inchannels=>embed_dims[1], stride=4, pad=Flux.SamePad()), 
        img2seq, 
        [SwinLayer(embed_dims[i], resolutions[i], depths[i], nheads[i], windows[i], shift_windows, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, i < nlayers) for i in 1:nlayers]..., 
        Flux.LayerNorm(embed_dims[4]), 
        seq2img, 
        Flux.AdaptiveMeanPool((1,1)), 
        Flux.MLUtils.flatten, 
        Flux.Dense(embed_dims[4] => nclasses)
    )
end

function SWIN(config::Symbol; kw...)
    @match config begin
        :tiny => swin(
            [96,192,384,768], 
            [2,2,6,2], 
            [3,6,12,24];
            kw...)
        :small => swin(
            [96,192,384,768], 
            [2,2,18,2], 
            [3,6,12,24];
            kw...)
        :base => swin(
            [128,256,512,1024], 
            [2,2,18,2], 
            [4,8,16,32];
            kw...)
        :large => swin(
            [192,384,768,1536], 
            [2,2,18,2], 
            [6,12,24,48];
            kw...)
    end
end