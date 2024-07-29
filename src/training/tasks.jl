abstract type AbstractTask end

struct BinarySegmentation{M} <: AbstractTask
    model::M
end

Flux.@layer BinarySegmentation

input(x::BinarySegmentation) = input(x.model)
encoder(x::BinarySegmentation) = encoder(x.model)
decoder(x::BinarySegmentation) = decoder(x.model)
head(x::BinarySegmentation) = head(x.model)

(m::BinarySegmentation)(x...) = m.model(x...) |> Flux.sigmoid

predict(m::BinarySegmentation, x...) = round.(Int, m(x...))