function SeparableConv(filter, in, out, σ=Flux.relu)
    Flux.Chain(
        Flux.DepthwiseConv(filter, in=>in, pad=Flux.SamePad()), 
        Flux.Conv((1,1), in=>out, σ, pad=Flux.SamePad())
    )
end

function ConvBlock(filter, in, out, σ; depth=2, batch_norm=false)
    return Flux.Chain(
        [ifelse(
            i == 1, 
            _conv(filter, in, out, σ, batch_norm), 
            _conv(filter, out, out, σ, batch_norm)
            ) 
        for i in 1:depth]...
    )
end

function _conv(filter, in, out, σ, bn)
    if bn
        return Flux.Chain(
            Flux.Conv(filter, in=>out, pad=Flux.SamePad()),
            Flux.BatchNorm(out), 
            σ
        )
    else
        return Flux.Conv(filter, in=>out, σ, pad=Flux.SamePad())
    end
end