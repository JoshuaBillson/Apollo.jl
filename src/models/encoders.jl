"""
    AbstractEncoder{F}

Super type of all encoder models. The type parameter `F` denotes the number of features
produced by each block of the encoder.

# Example Implementation
```julia
struct ResNet18{W} <: AbstractEncoder{(64,64,128,256,512)}
    weights::W
    ResNet18(;weights=nothing) = new{typeof(weights)}(weights)
end

build_encoder(x::ResNet18) = resnet(18, x.weights)
```
"""
abstract type AbstractEncoder{F} end

filters(::AbstractEncoder{F}) where {F} = F

"""
    build_encoder(encoder::AbstractEncoder)

Constructs an encoder model based on the provided `AbstractEncoder` configuration.

# Parameters
- `encoder`: An `AbstractEncoder` object specifying the architecture and configuration of the encoder to be built.

# Returns
A standard `Flux.Chain` layer containing each block of the encoder.
The returned encoder is ready to be integrated into more complex architectures like U-Net or used as a standalone feature extractor.

# Example
```julia
encoder = build_encoder(ResNet50(weights=:ImageNet))
```
"""
function build_encoder end

# ResNet18

"""
    ResNet18(;weights=nothing)

Construct a ResNet18 encoder with the specified initial weights.

# Parameters
- `weights`: Either `:ImageNet` or `nothing`.
"""
struct ResNet18{W} <: AbstractEncoder{(64,64,128,256,512)}
    weights::W
    ResNet18(;weights=nothing) = new{typeof(weights)}(weights)
end

build_encoder(e::ResNet18) = resnet(18, e.weights)

# ResNet34

"""
    ResNet34(;weights=nothing)

Construct a ResNet34 encoder with the specified initial weights.

# Parameters
- `weights`: Either `:ImageNet` or `nothing`.
"""
struct ResNet34{W} <: AbstractEncoder{(64,64,128,256,512)}
    weights::W
    ResNet34(;weights=nothing) = new{typeof(weights)}(weights)
end

build_encoder(e::ResNet34) = resnet(34, e.weights)

# ResNet50

"""
    ResNet50(;weights=nothing)

Construct a ResNet50 encoder with the specified initial weights.

# Parameters
- `weights`: Either `:ImageNet` or `nothing`.
"""
struct ResNet50{W} <: AbstractEncoder{(64,256,512,1024,2048)}
    weights::W
    ResNet50(;weights=nothing) = new{typeof(weights)}(weights)
end

build_encoder(e::ResNet50) = resnet(50, e.weights)

# ResNet101

"""
    ResNet101(;weights=nothing)

Construct a ResNet101 encoder with the specified initial weights.

# Parameters
- `weights`: Either `:ImageNet` or `nothing`.
"""
struct ResNet101{W} <: AbstractEncoder{(64,256,512,1024,2048)}
    weights::W
    ResNet101(;weights=nothing) = new{typeof(weights)}(weights)
end

build_encoder(e::ResNet101) = resnet(101, e.weights)

# ResNet152

"""
    ResNet152(;weights=nothing)

Construct a ResNet152 encoder with the specified initial weights.

# Parameters
- `weights`: Either `:ImageNet` or `nothing`.
"""
struct ResNet152{W} <: AbstractEncoder{(64,256,512,1024,2048)}
    weights::W
    ResNet152(;weights=nothing) = new{typeof(weights)}(weights)
end

build_encoder(e::ResNet152) = resnet(152, e.weights)

# UNet Encoder

"""
    StandardEncoder(;batch_norm=true)

Construct a standard `UNet` encoder with no pre-trained weights.
"""
struct StandardEncoder <: AbstractEncoder{(64,128,256,512,1024)}
    batch_norm::Bool
end

StandardEncoder(;batch_norm=true) = StandardEncoder(batch_norm)

function build_encoder(e::StandardEncoder)
    Flux.Chain(
        Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 64, 128, Flux.relu, batch_norm=e.batch_norm)), 
        Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 128, 256, Flux.relu, batch_norm=e.batch_norm)), 
        Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 256, 512, Flux.relu, batch_norm=e.batch_norm)), 
        Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 512, 1024, Flux.relu, batch_norm=e.batch_norm)), 
    )
end

# Base ResNet Constructor
resnet(depth, ::Nothing) = resnet(depth, :nothing)
function resnet(depth, weights::Symbol)
    # Construct Backbone
    backbone = @match weights begin
        :nothing => Metalhead.ResNet(depth, pretrain=false) |> Metalhead.backbone
        :ImageNet => Metalhead.ResNet(depth, pretrain=true) |> Metalhead.backbone
        _ => throw(ArgumentError("Weights must be one of :ImageNet or :nothing!"))
    end

    # Build Encoder
    return Flux.Chain(
        Flux.Chain(Flux.MaxPool((2,2)), backbone[2]...), 
        backbone[3], 
        backbone[4], 
        backbone[5], 
    )
end