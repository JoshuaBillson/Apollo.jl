"""
    Classifier(;input=RasterInput(), encoder=ResNet50(), nclasses=2)

Construct an image classifier from the provided encoder.
"""
struct Classifier{I,E,H}
    input::I
    encoder::E
    head::H
end

Flux.@layer Classifier

function Classifier(;input=RasterInput(), encoder=ResNet50(), nclasses=2)
    return Classifier(
        build_input(input, filters(encoder)[1]), 
        build_encoder(encoder), 
        Flux.Chain(Flux.GlobalMeanPool(), Flux.flatten, Flux.Dense(filters(encoder)[end]=>nclasses))
    )
end

(m::Classifier)(x) = @pipe m.input(x) |> m.encoder |> m.head |> Flux.softmax