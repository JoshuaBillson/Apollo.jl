abstract type AbstractEncoder{F1,F2,F3,F4,F5} end

# ResNet18

struct ResNet18{B} <: AbstractEncoder{64,64,128,256,512}
    backbone::B
end

Flux.@layer ResNet18

ResNet18(;weights=nothing) = ResNet18(resnet(18, weights))

(m::ResNet18)(x) = (x, Flux.activations(m.backbone, x)...)

# ResNet34

struct ResNet34{B} <: AbstractEncoder{64,64,128,256,512}
    backbone::B
end

Flux.@layer ResNet34

ResNet34(;weights=nothing) = ResNet34(resnet(34, weights))

(m::ResNet34)(x) = (x, Flux.activations(m.backbone, x)...)

# ResNet50

struct ResNet50{B} <: AbstractEncoder{64,256,512,1024,2048}
    backbone::B
end

Flux.@layer ResNet50

ResNet50(;weights=nothing) = ResNet50(resnet(50, weights))

(m::ResNet50)(x) = (x, Flux.activations(m.backbone, x)...)

# ResNet101

struct ResNet101{B} <: AbstractEncoder{64,256,512,1024,2048}
    backbone::B
end

Flux.@layer ResNet101

ResNet101(;weights=nothing) = ResNet101(resnet(101, weights))

(m::ResNet101)(x) = (x, Flux.activations(m.backbone, x)...)

# ResNet152

struct ResNet152{B} <: AbstractEncoder{64,256,512,1024,2048}
    backbone::B
end

Flux.@layer ResNet152

ResNet152(;weights=nothing) = ResNet152(resnet(152, weights))

(m::ResNet152)(x) = (x, Flux.activations(m.backbone, x)...)

# UNet Encoder

struct StandardEncoder{B} <: AbstractEncoder{64,128,256,512,1024}
    backbone::B
end

Flux.@layer StandardEncoder

function StandardEncoder(;batch_norm=true)
    StandardEncoder(
        Flux.Chain(
            Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 64, 128, Flux.relu, batch_norm=batch_norm)), 
            Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 128, 256, Flux.relu, batch_norm=batch_norm)), 
            Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 256, 512, Flux.relu, batch_norm=batch_norm)), 
            Flux.Chain(Flux.MaxPool((2,2)), ConvBlock((3,3), 512, 1024, Flux.relu, batch_norm=batch_norm)), 
        )
    )
end

(m::StandardEncoder)(x) = (x, Flux.activations(m.backbone, x)...)

# Base ResNet Constructor

resnet(depth, ::Nothing) = resnet(depth, false)
resnet(depth, weights::Symbol) = resnet(depth, weights==:ImageNet)
function resnet(depth, pretrain::Bool)
    backbone = Metalhead.ResNet(depth, pretrain=pretrain) |> Metalhead.backbone
    return Flux.Chain(
        Flux.Chain(Flux.MaxPool((2,2)), backbone[2]...), 
        backbone[3], 
        backbone[4], 
        backbone[5], 
    )
end

function filters(::AbstractEncoder{F1,F2,F3,F4,F5}) where {F1,F2,F3,F4,F5}
    return [F1,F2,F3,F4,F5]
end