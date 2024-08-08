"""
Super type of all input layers. Objects implementing this interface need to provide
an instance of the `build_input` method.

# Example Implementation
```julia
struct Single <: AbstractInput
    channels::Int
    batch_norm::Bool
end

Single(;channels=3, batch_norm=true) = Single(channels, batch_norm)

function build_input(x::Single, out_features::Int)
    ConvBlock((3,3), x.channels, out_features, Flux.relu, batch_norm=x.batch_norm)
end
```
"""
abstract type AbstractInput end

"""
    build_input(x::AbstractInput, out_features::Int)

Construct the given `AbstractInput` with the specified number of output features.
"""
function build_input end

"""
    Series(;channels=3, batch_norm=true)

Defines an input as a series of images, where each image contains the same number of `channels`.
The temporal features will first be extracted by a pixel-wise LSTM module, which will then be
subjected to two consecutive 3x3 convolutions.

# Parameters
- `channels`: The number of bands in each image.
- `batch_norm`: If `true`, batch normalization will be applied after the convolutional layers.
"""
struct Series <: AbstractInput
    channels::Int
    batch_norm::Bool
end

Series(;channels=3, batch_norm=true) = Series(channels, batch_norm)

function build_input(x::Series, out_features::Int)
    Flux.Chain(
        LSTM(x.channels => out_features), 
        ConvBlock((3,3), out_features, out_features, Flux.relu, batch_norm=x.batch_norm)
    )
end

"""
    Single(;channels=3, batch_norm=true)

Defines an input as a singular image. Low-level features are extracted by two consecutive
3x3 convolutions.

# Parameters
- `channels`: The number of bands in the image.
- `batch_norm`: If `true`, batch normalization will be applied after the convolutional layers.
"""
struct Single <: AbstractInput
    channels::Int
    batch_norm::Bool
end

Single(;channels=3, batch_norm=true) = Single(channels, batch_norm)

function build_input(x::Single, out_features::Int)
    ConvBlock((3,3), x.channels, out_features, Flux.relu, batch_norm=x.batch_norm)
end