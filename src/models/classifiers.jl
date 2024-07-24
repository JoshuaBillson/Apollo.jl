struct Classifier{I,E,H}
    input::I
    encoder::E
    head::H
end

Flux.@layer Classifier

function Classifier(encoder::AbstractEncoder; channels=3, nclasses=1, batch_norm=true)
    Classifier(
        ConvBlock((3,3), channels, filters(encoder)[1], Flux.relu, batch_norm=batch_norm), 
        encoder, 
        Flux.Chain( Flux.GlobalMeanPool(), Flux.flatten, Flux.Dense(filters(encoder)[end]=>nclasses) )
    )
end

(m::Classifier)(x) = @pipe m.input(x) |> m.encoder |> last |> m.head