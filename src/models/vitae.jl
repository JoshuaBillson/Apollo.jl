function PRM(kernel_size, in_features, embedding, stride, dilations)
    kernel = (kernel_size, kernel_size)
    filters = in_features => embedding
    Flux.Parallel(
        (x...) -> cat(x..., dims=3), 
        [Flux.Conv(kernel, filters, Flux.gelu, stride=stride, pad=Flux.SamePad(), dilation=d) for d in dilations]...
    )
end

function PCM(in_features, embedding, downsample, groups)
    strides = [(downsample - 2 * i) > 0 ? 2 : 1 for i in 0:2]
    Flux.Chain(
        Flux.Conv((3,3), in_features => embedding, pad=Flux.SamePad(), groups=groups, stride=strides[1]), 
        Flux.BatchNorm(embedding, Flux.swish),
        Flux.Conv((3,3), embedding => embedding, pad=Flux.SamePad(), groups=groups, stride=strides[2]), 
        Flux.BatchNorm(embedding, Flux.swish), 
        Flux.Conv((3,3), embedding => embedding, pad=Flux.SamePad(), groups=groups, stride=strides[3]), 
        Flux.BatchNorm(embedding, Flux.swish), 
    )
end