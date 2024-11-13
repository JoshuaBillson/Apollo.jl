struct SRAttention{N,S,Q,K,P,D1,D2}
    nheads::Int
    norm::N
    sr_layer::S
    q_layer::Q
    kv_layer::K
    projection::P
    attn_drop::D1
    proj_drop::D2
end

Flux.@layer :expand SRAttention

function SRAttention(dim::Int; nheads::Int = 8, qkv_bias::Bool = false, attn_dropout_prob = 0.0, proj_dropout_prob = 0.0, sr_ratio=1)
    return SRAttention(
        nheads,
        Flux.LayerNorm(dim), 
        sr_ratio == 1 ? nothing : Flux.Conv((sr_ratio,sr_ratio), dim => dim, stride=sr_ratio), 
        Flux.Dense(dim, dim; bias = qkv_bias), 
        Flux.Dense(dim, dim*2; bias = qkv_bias), 
        Flux.Dense(dim, dim), 
        Flux.Dropout(attn_dropout_prob), 
        Flux.Dropout(proj_dropout_prob), 
    )
end

function (m::SRAttention)(x)
    q = m.q_layer(x)
    sr = isnothing(m.sr_layer) ? x : seq2img(x) |> m.sr_layer |> img2seq |> m.norm
    kv = m.kv_layer(sr)
    k, v = Flux.chunk(kv, 2, dims = 1)
    y, α = Flux.NNlib.dot_product_attention(q, k, v; m.nheads, fdrop = m.attn_drop)
    y = m.projection(y) |> m.proj_drop
    return y
end

function PVTEncoderBlock(dim, nheads; mlp_ratio=4, qkv_bias=false, drop=0., attn_drop=0., drop_path=0., sr_ratio=1)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                SRAttention(dim; nheads, qkv_bias, sr_ratio, attn_dropout_prob=attn_drop, proj_dropout_prob=drop),
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

function PVTBlock(in_planes, out_planes; nheads=8, imsize=224, patch_size=16, depth=4, mlp_ratio=4, qkv_bias=false, drop=0., attn_drop=0., drop_path=0., sr_ratio=1)
    n_patches = (imsize ÷ patch_size) ^ 2
    Flux.Chain(
        Flux.Conv((patch_size,patch_size), in_planes=>out_planes, stride=patch_size), 
        img2seq, 
        Flux.LayerNorm(out_planes), 
        Metalhead.Layers.ViPosEmbedding(out_planes, n_patches), 
        Flux.Dropout(drop), 
        [PVTEncoderBlock(out_planes, nheads; mlp_ratio, qkv_bias, drop, attn_drop, drop_path, sr_ratio) for _ in 1:depth]..., 
        seq2img
    )
end

function pvt(embed_dims, depths, nheads, mlp_ratios, sr_ratios; 
    qkv_bias=true, drop=0.1, attn_drop=0.1, nclasses=1000, imsize=224, inchannels=3)

    in_dims = vcat([inchannels], embed_dims[1:end-1])
    img_sizes = vcat([imsize], [imsize ÷ (2 ^ (i + 1)) for i in eachindex(depths)])[1:end-1]
    Flux.Chain(
        Flux.Chain(
            [PVTBlock(
                in_dims[i], 
                embed_dims[i],
                nheads=nheads[i], 
                imsize=img_sizes[i], 
                patch_size=(i == 1 ? 4 : 2), 
                depth=depths[i], 
                mlp_ratio=mlp_ratios[i], 
                sr_ratio=sr_ratios[i], 
                qkv_bias=qkv_bias, 
                drop=drop, 
                attn_drop=attn_drop, 
            ) for i in eachindex(depths)]...,
        ), 
        Flux.AdaptiveMeanPool((1,1)), 
        Flux.MLUtils.flatten, 
        Flux.Dense(embed_dims[end] => nclasses)
    )
end

function PVT(config::Symbol; kw...)
    @match config begin
        :tiny => pvt(
            [64,128,320,512], # Embed Dims
            [2,2,2,2],        # Block Depths
            [1,2,5,8],        # Number of Heads
            [8,8,4,4],        # MLP Ratio
            [8,4,2,1];        # SR Ratios
            kw...)
        :small => pvt(
            [64,128,320,512], # Embed Dims
            [3,4,6,3],        # Block Depths
            [1,2,5,8],        # Number of Heads
            [8,8,4,4],        # MLP Ratio
            [8,4,2,1];        # SR Ratios
            kw...)
        :medium => pvt(
            [64,128,320,512], # Embed Dims
            [3,4,18,3],       # Block Depths
            [1,2,5,8],        # Number of Heads
            [8,8,4,4],        # MLP Ratio
            [8,4,2,1];        # SR Ratios
            kw...)
        :large => pvt(
            [64,128,320,512], # Embed Dims
            [3,8,27,3],       # Block Depths
            [1,2,5,8],        # Number of Heads
            [8,8,4,4],        # MLP Ratio
            [8,4,2,1];        # SR Ratios
            kw...)
    end
end