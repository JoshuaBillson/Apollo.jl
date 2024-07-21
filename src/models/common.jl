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
            Conv(filter, in, out, σ, batch_norm=batch_norm), 
            Conv(filter, out, out, σ, batch_norm=batch_norm)
            ) 
        for i in 1:depth]...
    )
end

function Conv(filter, in, out, σ; batch_norm=false)
    if batch_norm
        return Flux.Chain(
            Flux.Conv(filter, in=>out, pad=Flux.SamePad()),
            Flux.BatchNorm(out, σ), 
        )
    else
        return Flux.Conv(filter, in=>out, σ, pad=Flux.SamePad())
    end
end