struct ConvAttention{Q,K,V,P,D1,D2}
    nheads::Int
    q_layer::Q
    k_layer::K
    v_layer::V
    projection::P
    attn_drop::D1
    proj_drop::D2
end

Flux.@layer :expand ConvAttention

function ConvAttention(dim::Int; nheads::Int = 8, attn_dropout_prob = 0.0, proj_dropout_prob = 0.0, q_stride=1, k_stride=1, v_stride=1)
    return ConvAttention(
        nheads,
        SeparableConv((3,3), dim, dim, stride=q_stride), 
        SeparableConv((3,3), dim, dim, stride=k_stride), 
        SeparableConv((3,3), dim, dim, stride=v_stride), 
        Flux.Dense(dim, dim),
        Flux.Dropout(attn_dropout_prob), 
        Flux.Dropout(proj_dropout_prob), 
    )
end

function (m::ConvAttention)(x::AbstractArray{<:Real,3})
    x = seq2img(x)
    q = m.q_layer(x) |> img2seq
    k = m.k_layer(x) |> img2seq
    v = m.v_layer(x) |> img2seq
    y, α = Flux.NNlib.dot_product_attention(q, k, v; m.nheads, fdrop = m.attn_drop)
    return m.projection(y) |> m.proj_drop
end

function CVTTransformer(dim; nheads=1, mlp_ratio=4, dropout=0.)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim), 
                ConvAttention(dim; nheads, k_stride=2, v_stride=2, attn_dropout_prob=dropout, proj_dropout_prob=dropout)
            ), 
            +
        ), 
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim), 
                MLP(dim, dim * mlp_ratio, dim, dropout)
            ), 
            +
        )
    )
end

function CVTBlock(dim, depth::Int; nheads=1, mlp_ratio=4, dropout=0.)
    Flux.Chain([CVTTransformer(dim; nheads, mlp_ratio, dropout) for _ in 1:depth]...)
end

function CVT(;inchannels=3, nclasses=1000, dim=64, nheads=[1,3,6], strides=[4,2,2], depths=[1,2,10], dropout=0., mlp_ratio=4)
    Flux.Chain(

        # Encoder
        Flux.Chain(

            # Stem
            Flux.Conv((7,7), inchannels=>dim, stride=strides[1], pad=Flux.SamePad()), 

            # Stage 1
            Flux.Chain(
                img2seq, 
                Flux.LayerNorm(dim), 
                CVTBlock(dim, depths[1]; nheads=nheads[1], dropout, mlp_ratio), 
                seq2img, 
            ), 

            # Stage 2
            Flux.Chain(
                Flux.Conv((3,3), dim=>dim*nheads[2], stride=strides[2], pad=Flux.SamePad()), 
                img2seq, 
                Flux.LayerNorm(dim*nheads[2]), 
                CVTBlock(dim*nheads[2], depths[2]; nheads=nheads[2], dropout, mlp_ratio), 
                seq2img, 
            ), 

            # Stage 3
            Flux.Chain(
                Flux.Conv((3,3), dim*nheads[2]=>dim*nheads[3], stride=strides[3], pad=Flux.SamePad()), 
                img2seq, 
                Flux.LayerNorm(dim*nheads[3]), 
                CVTBlock(dim*nheads[3], depths[3]; nheads=nheads[3], dropout, mlp_ratio), 
                seq2img, 
            ), 
        ), 

        # Head
        Flux.Chain(
            Flux.GlobalMeanPool(), 
            x -> dropdims(x; dims=(1,2)), 
            Flux.LayerNorm(dim*nheads[3]), 
            Flux.Dense(dim*nheads[3]=>nclasses)
        )
    )
end