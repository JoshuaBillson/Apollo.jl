struct ResNet{I,B,H}
    input::I
    backbone::B
    head::H
end

Flux.@layer ResNet

function ResNet(depth::Int; pretrain=false, channels=3, nclasses=1000)
    base = Metalhead.ResNet(depth, pretrain=false, inchannels=channels, nclasses=nclasses)
    if pretrain
        pretrained = Metalhead.ResNet(depth, pretrain=true, inchannels=3, nclasses=nclasses)
        input = Metalhead.backbone(base)[1]
        backbone = Metalhead.backbone(pretrained)[2:end]
        head = Metalhead.classifier(pretrained)
        return ResNet(input, backbone, head)
    else
        input = Metalhead.backbone(base)[1]
        backbone = Metalhead.backbone(base)[2:end]
        head = Metalhead.classifier(base)
        return ResNet(input, backbone, head)
    end
end

function activations(m::ResNet, x)
    input_out = m.input(x)
    backbone_out = Flux.activations(m.backbone, input_out)
    head_out = last(backbone_out) |> m.head
    return [input_out, backbone_out..., head_out]
end

function (m::ResNet)(x)
    return m.input(x) |> m.backbone |> m.head
end