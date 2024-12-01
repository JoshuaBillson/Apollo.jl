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
    if patch_merging
        return Flux.Chain(img2seq, PatchMerging(dim ÷ 2), blocks..., seq2img)
    end
    return Flux.Chain(img2seq, blocks..., seq2img)
end

function swin(embed_dims, depths, nheads; 
    inchannels=3, nclasses=1000, windows=[7,7,7,7], mlp_ratio=4, qkv_bias=true, drop_rate=0.1, attn_drop_rate=0.1, shift_windows=true)
    nlayers = length(depths)
    resolutions = [(224 ÷ 4) ÷ (2 ^ (i-1)) for i in eachindex(depths)]
    Flux.Chain(
        Flux.Chain(
            Flux.Conv((4,4), inchannels=>embed_dims[1], stride=4, pad=Flux.SamePad()), 
            [SwinLayer(embed_dims[i], resolutions[i], depths[i], nheads[i], windows[i], shift_windows, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, i != 1) for i in 1:nlayers]..., 
            img2seq, 
            Flux.LayerNorm(embed_dims[4])
        ), 
        Flux.Chain(
            seq2img, 
            Flux.AdaptiveMeanPool((1,1)), 
            Flux.MLUtils.flatten, 
            Flux.Dense(embed_dims[4] => nclasses)
        )
    )
end

function swin_unet_block(dim::Int, imsize::Int, nheads::Int; window_size=7)
    window_shift = window_size ÷ 2
    blocks = [
        SwinTransformerBlock(
            dim, 
            nheads; 
            imsize=imsize,
            window_size=window_size, 
            mlp_ratio=4, 
            qkv_bias=true, 
            drop=0.1, 
            attn_drop=0.1, 
            window_shift = ((i % 2) == 0) ? window_shift : 0) 
    for i in 1:2]
    return Flux.Chain(blocks...)
end

function swin_unet_decoder(dim::Int, imsize::Int, nheads::Int, expand_patches::Bool; window_size=7)
    window_shift = window_size ÷ 2
    blocks = [
        SwinTransformerBlock(
            dim, 
            nheads; 
            imsize=imsize,
            window_size=window_size, 
            mlp_ratio=4, 
            qkv_bias=true, 
            drop=0.1, 
            attn_drop=0.1, 
            window_shift = ((i % 2) == 0) ? window_shift : 0) 
    for i in 1:2]
    return expand_patches ? Flux.Chain(img2seq, PatchExpanding(dim * 2), blocks..., seq2img) : Flux.Chain(img2seq, blocks..., seq2img)
end

function swin_unet(;inchannels=3, nclasses=3, imsize=224)
    feature_sizes = [(imsize ÷ 4) ÷ (2 ^ (i-1)) for i in 1:4]
    Flux.Chain(
        # Patch Embedding
        Flux.Conv((4,4), inchannels=>128, stride=4, pad=Flux.SamePad()), 

        # Encoder
        Flux.Chain(

            # Encoder Block 1
            img2seq, 
            swin_unet_block(128, feature_sizes[1], 4), 
            Flux.SkipConnection(
                Flux.Chain(

                    # Encoder Block 2
                    PatchMerging(128), 
                    swin_unet_block(256, feature_sizes[2], 8), 
                    Flux.SkipConnection(
                        Flux.Chain(

                            # Encoder Block 3
                            PatchMerging(256), 
                            swin_unet_block(512, feature_sizes[3], 16), 
                            Flux.SkipConnection(

                                # Bottle-Neck
                                Flux.Chain(
                                    PatchMerging(512), 
                                    swin_unet_block(1024, feature_sizes[4], 32), 
                                    PatchExpanding(1024)
                                ), 
                                (a, b) -> cat(a, b, dims=1)
                            ), 

                            # Decoder Block 3
                            Flux.Dense(1024=>512), 
                            swin_unet_block(512, feature_sizes[3], 16), 
                            PatchExpanding(512)
                        ), 
                        (a, b) -> cat(a, b, dims=1)
                    ), 

                    # Decoder Block 2
                    Flux.Dense(512=>256), 
                    swin_unet_block(256, feature_sizes[2], 8), 
                    PatchExpanding(256)
                ), 
                (a, b) -> cat(a, b, dims=1)
            ), 

            # Decoder Block 1
            Flux.Dense(256=>128), 
            swin_unet_block(128, feature_sizes[1], 4),
            seq2img, 
            Base.Fix2(Flux.upsample_bilinear, (4,4)), 
            Flux.Conv((3,3), 128=>128, Flux.relu, pad=Flux.SamePad()), 
        ), 

        # Classification Head
        Flux.Conv((1,1), 128=>nclasses)
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