function SeparableConv(filter, in, out, σ=Flux.relu)
    Flux.Chain(
        Flux.DepthwiseConv(filter, in=>in, pad=Flux.SamePad()), 
        Flux.Conv((1,1), in=>out, σ, pad=Flux.SamePad())
    )
end

function ConvBlock(filter, in, out, σ; depth=2, batch_norm=true)
    return Flux.Chain(
        [ifelse(
            i == 1, 
            Conv(filter, in, out, σ, batch_norm=batch_norm), 
            Conv(filter, out, out, σ, batch_norm=batch_norm)
            ) 
        for i in 1:depth]...
    )
end

function Conv(filter, in, out, σ; batch_norm=true)
    if batch_norm
        return Flux.Chain(
            Flux.Conv(filter, in=>out, pad=Flux.SamePad()),
            Flux.BatchNorm(out, σ), 
        )
    else
        return Flux.Conv(filter, in=>out, σ, pad=Flux.SamePad())
    end
end

struct LSTM{T}
    lstm::T
end

Flux.@layer LSTM

function LSTM(in_out::Pair{Int,Int})
    return LSTM(Flux.LSTM(in_out[1] => in_out[2]))
end

function (m::LSTM)(x::AbstractArray{<:AbstractFloat,5})
    # Reset LSTM State
    Flux.reset!(m.lstm)

    # Run Forward Pass
    @pipe permutedims(x, (3, 1, 2, 4, 5)) |>             # Permute to (CWHLN)
    [selectdim(_, 4, i) for i in 1:9] |>                 # Split Time Series
    [reshape(x, (2,:)) for x in _] |>                    # Reshape to (CN)
    [m.lstm(x) for x in _] |>                            # Apply LSTM to each time stamp
    last |>                                              # Keep Last Prediction
    reshape(_, (:, size(x,1), size(x,2), size(x,5))) |>  # Reshape to (CWHN)
    permutedims(_, (2, 3, 1, 4))                         # Permute to (WHCN)
end