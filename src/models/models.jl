struct BinarySegmentationModel{M}
    model::M
end

Flux.@layer BinarySegmentationModel

(m::BinarySegmentationModel)(x...) = m.model(x...) |> Flux.sigmoid