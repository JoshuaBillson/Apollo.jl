struct ResNet{I,B,H}
    input::I
    backbone::B
    head::H
end

Flux.@layer ResNet

function ResNet(depth::Int; pretrain=false, channels=3, nclasses=1000)
    if pretrain
        pretrained = Metalhead.ResNet(depth, pretrain=true, inchannels=3, nclasses=nclasses)
        input = ConvBlock((3,3), channels, 64, Flux.relu, batch_norm=true)
        #input = Flux.Chain(Flux.Conv((7, 7), channels => 64, pad=3, stride=2, bias=false), Flux.BatchNorm(64, Flux.relu))
        #backbone = Flux.Chain(Flux.Chain(Flux.MaxPool((3, 3), pad=1, stride=2), p_backbone[2]...), p_backbone[3:end]...)
        backbone_1 = Flux.Chain(Flux.MaxPool((2,2)), Metalhead.backbone(pretrained)[2]...)
        backbone = Flux.Chain(backbone_1,  Metalhead.backbone(pretrained)[3:end]...)
        head = Metalhead.classifier(pretrained)
        return ResNet(input, backbone, head)
    else
        base = Metalhead.ResNet(depth, pretrain=false, inchannels=channels, nclasses=nclasses)
        input = ConvBlock((3,3), channels, 64, Flux.relu, batch_norm=true)
        backbone = Metalhead.backbone(base)[2:end]
        #input = Metalhead.backbone(base)[1][1:2]
        #backbone = Flux.Chain(Flux.Chain(Flux.MaxPool((3, 3), pad=1, stride=2), p_backbone[2]...), p_backbone[3:end]...)
        head = Metalhead.classifier(base)
        return ResNet(input, backbone, head)
    end
end

function activations(m::ResNet, x)
    input_out = m.input(x)
    backbone_out = Flux.activations(m.backbone, input_out)
    head_out = last(backbone_out) |> m.head
    return Tuple([input_out, backbone_out..., head_out])
end

function (m::ResNet)(x)
    return m.input(x) |> m.backbone |> m.head
end