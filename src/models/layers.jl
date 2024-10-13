"""
    SeparableConv(filter, in, out, σ=Flux.relu)

A separable convolutional layer consists of a depthwise convolution followed by a 1x1 convolution.

# Parameters
- `filter`: The size of the filter as a tuple.
- `in`: The number of input features.
- `out`: The number of output features.
- `σ`: The activation function to apply following the 1x1 convolution.
"""
function SeparableConv(kernel_size::Int, in::Int, out::Int, σ=Flux.relu)
    Flux.Chain(
        Flux.DepthwiseConv((kernel_size, kernel_size), in=>in, pad=Flux.SamePad()), 
        Flux.Conv((1,1), in=>out, σ, pad=Flux.SamePad())
    )
end

"""
    ConvBlock(filter, in, out, σ; depth=2, batch_norm=true)

A block of convolutional layers with optional batch normalization.

# Parameters
- `filter`: The size of the filter to use for each layer.
- `in`: The number of input features.
- `out`: The number of output features.
- `σ`: The activation function to apply after each convolution.
- `depth`: The number of successive convolutions in the block.
- `batch_norm`: Applies batch normalization after each convolution if `true`.
"""
function ConvBlock(kernel_size::Int, in::Int, out::Int, σ; depth=2, batch_norm=true)
    return Flux.Chain(
        [ifelse(
            i == 1, 
            Conv(kernel_size, in, out, σ, batch_norm=batch_norm), 
            Conv(kernel_size, out, out, σ, batch_norm=batch_norm)
            ) 
        for i in 1:depth]...
    )
end

function Conv(kernel_size::Int, in::Int, out::Int, σ; batch_norm=true, groups=1, dilation=1)
    if batch_norm
        return Flux.Chain(
            Flux.Conv((kernel_size, kernel_size), in=>out, pad=Flux.SamePad(), groups=groups, dilation=dilation),
            Flux.BatchNorm(out, σ), 
        )
    else
        return Flux.Conv((kernel_size, kernel_size), in=>out, σ, pad=Flux.SamePad())
    end
end

"""
    LSTM(in => out)

A stateless pixel-wise LSTM layer that expects an input tensor with shape WHCLN, where
W is image width, H is image height, C is image channels, L is sequence length, and N is 
batch size. The LSTM module will be applied to each pixel in the sequence from first to last 
with all but the final output discarded.

# Parameters
- `in`: The number of input features.
- `out`: The number of output features.

# Example
```jldoctest
julia> m = Apollo.LSTM(2=>32);

julia> x = rand(Float32, 64, 64, 2, 9, 4);

julia> m(x) |> size
(64, 64, 32, 4)
```
"""
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
    [_[:,:,:,i,:] for i in 1:size(x, 4)] |>              # Split Time Series
    [reshape(x, (2,:)) for x in _] |>                    # Reshape to (CN)
    [m.lstm(x) for x in _] |>                            # Apply LSTM to each time stamp
    last |>                                              # Keep Last Prediction
    reshape(_, (:, size(x,1), size(x,2), size(x,5))) |>  # Reshape to (CWHN)
    permutedims(_, (2, 3, 1, 4))                         # Permute to (WHCN)
end
