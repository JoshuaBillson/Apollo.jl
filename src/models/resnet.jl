struct ResNet18{C2,C3,C4,C5}
    conv2::C2
    conv3::C3
    conv4::C4
    conv5::C5
end

Flux.@layer ResNet18

ResNet18(;pretrain=false) = resnet(ResNet18, 18, pretrain)

filters(::Type{<:ResNet18}) = [64, 64, 128, 256, 512]

function (m::ResNet18)(x)
    x1 = m.conv2(x)
    x2 = m.conv3(x1)
    x3 = m.conv4(x2)
    x4 = m.conv5(x3)
    return (x, x1, x2, x3, x4)
end

struct ResNet34{C2,C3,C4,C5}
    conv2::C2
    conv3::C3
    conv4::C4
    conv5::C5
end

Flux.@layer ResNet34

ResNet34(;pretrain=false) = resnet(ResNet34, 34, pretrain)

filters(::Type{<:ResNet34}) = [64, 64, 128, 256, 512]

function (m::ResNet34)(x)
    x1 = m.conv2(x)
    x2 = m.conv3(x1)
    x3 = m.conv4(x2)
    x4 = m.conv5(x3)
    return (x, x1, x2, x3, x4)
end

struct ResNet50{C2,C3,C4,C5}
    conv2::C2
    conv3::C3
    conv4::C4
    conv5::C5
end

Flux.@layer ResNet50

ResNet50(;pretrain=false) = resnet(ResNet50, 50, pretrain)

filters(::Type{<:ResNet50}) = [64, 256, 512, 1024, 2048]

function (m::ResNet50)(x)
    x1 = m.conv2(x)
    x2 = m.conv3(x1)
    x3 = m.conv4(x2)
    x4 = m.conv5(x3)
    return (x, x1, x2, x3, x4)
end

struct ResNet101{C2,C3,C4,C5}
    conv2::C2
    conv3::C3
    conv4::C4
    conv5::C5
end

Flux.@layer ResNet101

ResNet101(;pretrain=false) = resnet(ResNet101, 101, pretrain)

filters(::Type{<:ResNet101}) = [64, 256, 512, 1024, 2048]

function (m::ResNet101)(x)
    x1 = m.conv2(x)
    x2 = m.conv3(x1)
    x3 = m.conv4(x2)
    x4 = m.conv5(x3)
    return (x, x1, x2, x3, x4)
end

struct ResNet152{C2,C3,C4,C5}
    conv2::C2
    conv3::C3
    conv4::C4
    conv5::C5
end

Flux.@layer ResNet152

ResNet152(;pretrain=false) = resnet(ResNet152, 152, pretrain)

filters(::Type{<:ResNet152}) = [64, 256, 512, 1024, 2048]

function (m::ResNet152)(x)
    x1 = m.conv2(x)
    x2 = m.conv3(x1)
    x3 = m.conv4(x2)
    x4 = m.conv5(x3)
    return (x, x1, x2, x3, x4)
end

function resnet(::Type{T}, depth, pretrain) where {T}
    backbone = Metalhead.ResNet(depth, pretrain=pretrain) |> Metalhead.backbone
    return T(
        Flux.Chain(Flux.MaxPool((2,2)), backbone[2]...), 
        backbone[3], 
        backbone[4], 
        backbone[5], 
    )
end