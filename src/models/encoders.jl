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

# ResNet Encoder

struct ResNet
    depth::Int
    weights::Symbol
end

function ResNet(;depth=50, weights=:ImageNet)
    @argcheck depth in (18,34,50,101,152)
    @argcheck weights in (:Nothing,:ImageNet)
    return ResNet(depth, weights)
end

function filters(x::ResNet)
    @match x.depth begin
        18 => (64,64,128,256,512)
        34 => (64,64,128,256,512)
        50 => (64,256,512,1024,2048)
        101 => (64,256,512,1024,2048)
        152 => (64,256,512,1024,2048)
    end
end

function build_encoder(x::ResNet, inchannels::Int)
    # Validate Channels
    pattern = (x.weights, inchannels)
    input = @match pattern begin
        (:ImageNet, 3) => Metalhead.backbone(Metalhead.ResNet(x.depth, pretrain=true))[1][1:2]
        (:Nothing, c) => Metalhead.backbone(Metalhead.ResNet(x.depth, inchannels=c))[1][1:2]
        (:ImageNet, c) => begin
            @warn "ImageNet only supports 3 channels! Using uninitialized input."
            Metalhead.backbone(Metalhead.ResNet(x.depth, inchannels=c))[1][1:2]
        end
    end

    # Construct Backbone
    backbone = @match x.weights begin
        :Nothing => Metalhead.backbone(Metalhead.ResNet(x.depth, pretrain=false))[2:end]
        :ImageNet => Metalhead.backbone(Metalhead.ResNet(x.depth, pretrain=true))[2:end]
    end

    # Build Encoder
    return (
        input,
        Flux.Chain(Flux.MaxPool((2,2)), backbone[1]...), 
        backbone[2], 
        backbone[3], 
        backbone[4], 
    )
end

# UNet Encoder

"""
    StandardEncoder(;batch_norm=true)

Construct a standard `UNet` encoder with no pre-trained weights.
"""
struct StandardEncoder <: AbstractEncoder{(64,128,256,512,1024)}
    batch_norm::Bool
end

StandardEncoder(;batch_norm=true) = StandardEncoder(batch_norm)

filters(::StandardEncoder) = (64,128,256,512,1024)

function build_encoder(e::StandardEncoder)
    return (
        Flux.Chain(ConvBlock((3,3), 3, 64, Flux.relu, batch_norm=e.batch_norm)), 
        Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 64, 128, Flux.relu, batch_norm=e.batch_norm)), 
        Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 128, 256, Flux.relu, batch_norm=e.batch_norm)), 
        Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 256, 512, Flux.relu, batch_norm=e.batch_norm)), 
        Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 512, 1024, Flux.relu, batch_norm=e.batch_norm)), 
    )
end


struct Identity end

Flux.@layer Identity

(m::Identity)(x) = x