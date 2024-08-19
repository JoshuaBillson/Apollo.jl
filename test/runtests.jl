using Apollo
using Test
using Rasters
using Random
using StableRNGs

const rng = StableRNG(123)

@testset "tensor" begin
    # Test Data
    r1 = Raster(rand(rng, Float32, 256, 256, 3), (X, Y, Band))
    r2 = Raster(rand(rng, Float32, 3, 256, 256), (Band, X, Y))
    r3 = Raster(rand(rng, Float32, 128, 128, 3), (X, Y, Band))
    r4 = Raster(rand(rng, Float32, 128, 128, 3, 9), (X, Y, Band, Ti))
    r5 = Raster(rand(rng, Float32, 3, 128, 128, 9), (Band, X, Y, Ti))
    r6 = Raster(rand(rng, Float32, 256, 256), (X, Y))

    # tensor
    @test size(tensor(r1)) == (256, 256, 3, 1)  # simple case
    @test size(tensor(r1, r2)) == (256, 256, 3, 2)  # Mutiple rasters with different dim order
    @test size(tensor(r4)) == (128, 128, 3, 9, 1) 
    @test size(tensor(r5)) == (128, 128, 3, 9, 1)
    @test size(tensor(r4, r5)) == (128, 128, 3, 9, 2)
    @test size(tensor(r6, r6)) == (256, 256, 1, 2)  # Missing Bands
    @test size(tensor(Float64, [r1, r1, r2])) == (256, 256, 3, 3)  # Vector of rasters
    @test size(tensor(RasterStack(r1, layersfrom=Band), layerdim=Band)) == (256, 256, 3, 1)  # RasterStack
    @test_throws DimensionMismatch tensor(r1, r2, r4)  # Extra Dimension
    @test_throws DimensionMismatch tensor(r4, r5, r1)  # Missing Dimension
    @test_throws DimensionMismatch tensor(r1, r2, r6)  # Missing Dimension
    @test_throws DimensionMismatch tensor(r1, r2, r3)  # Mismatched Sizes

    # raster
    @test all(raster(tensor(r1), dims(r1)) .== r1)  # raster dims match tensor dims
    @test all(raster(tensor(r2), dims(r2)) .== permutedims(r2, (X,Y,Band)))  # raster dims mismatch tensor dims

    # resample
    @test size(Apollo.resample(r1, 2.0, :bilinear)) == (512, 512, 3)
    @test size(Apollo.resample(r1, 0.5, :average)) == (128, 128, 3)
    @test size(Apollo.resample(r5, 2.0, :bilinear)) == (3, 256, 256, 9)
    @test size(Apollo.resample(r5, 0.5, :average)) == (3, 64, 64, 9)
    @test_throws ArgumentError Apollo.resample(r5, 0.5, :foo)
end

@testset "utilities" begin
    # Test Data
    r1 = Raster(rand(rng, 256, 256), (X, Y))
    r2 = Raster(rand(rng, 256, 256, 3), (X, Y, Band))
    r3 = Raster(rand(rng, 256, 256, 3, 9), (X, Y, Band, Ti))
    r4 = mask(r2, with=Raster(rand(Bool, 256, 256, 3), (X,Y,Band), missingval=false))
    rs1 = RasterStack(r2, layersfrom=Band)
    rs2 = RasterStack(r3, layersfrom=Ti)
    rs3 = mask(rs1, with=Raster(rand(Bool, 256, 256), (X,Y), missingval=false))

    # catlayers
    @test size(catlayers(rs1, Band)) == (256,256,3)
    @test catlayers(rs1, Band) isa Raster
    @test size(catlayers(rs2, Ti)) == (256,256,3,9)
    @test catlayers(rs2, Ti) isa Raster
    @test size(catlayers(rs2, Band)) == (256,256,27)
    @test catlayers(rs2, Ti) isa Raster

    # foldlayers
    @test foldlayers(sum, rs1).Band_1 == sum(rs1[:Band_1])
    @test foldlayers(sum, rs1).Band_2 == sum(rs1[:Band_2])
    @test foldlayers(sum, rs1).Band_3 == sum(rs1[:Band_3])
    @test foldlayers(sum, rs3).Band_1 == (rs3[:Band_1] |> replace_missing |> vec |> skipmissing |> sum)
    @test foldlayers(sum, rs3).Band_2 == (rs3[:Band_2] |> replace_missing |> vec |> skipmissing |> sum)
    @test foldlayers(sum, rs3).Band_3 == (rs3[:Band_3] |> replace_missing |> vec |> skipmissing |> sum)

    # folddims
    @test folddims(sum, r3; dims=Band)[1] == sum(r3[Band(1)])
    @test folddims(sum, r3; dims=Band)[2] == sum(r3[Band(2)])
    @test folddims(sum, r3; dims=Band)[3] == sum(r3[Band(3)])
    @test folddims(sum, r4)[1] == (r4[Band(1)] |> replace_missing |> vec |> skipmissing |> sum)
    @test folddims(sum, r4)[2] == (r4[Band(2)] |> replace_missing |> vec |> skipmissing |> sum)
    @test folddims(sum, r4)[3] == (r4[Band(3)] |> replace_missing |> vec |> skipmissing |> sum)

    # putdim
    @test size(putdim(r1, Band)) == (256,256,1)
    @test hasdim(putdim(r1, Band), Band)
    @test size(putdim(r2, Ti)) == (256,256,3,1)
    @test hasdim(putdim(r2, Ti), Ti)
    @test putdim(r2, (X,Y,Band)) == r2
    @test size(putdim(r2, (Z, Band, Ti, X, Y))) == (256,256,3,1,1)
    @test hasdim(putdim(r1, (Z, Band, Ti, X, Y)), Z)
    @test hasdim(putdim(r1, (Z, Band, Ti, X, Y)), Band)
    @test hasdim(putdim(r1, (Z, Band, Ti, X, Y)), Ti)

    # ones_like
    @test size(ones_like(r3)) == (256,256,3,9)
    @test ones_like(r3) isa Array{Float64,4}
    @test all(==(1), ones_like(r3))

    # zeros_like
    @test size(zeros_like(r3)) == (256,256,3,9)
    @test zeros_like(r3) isa Array{Float64,4}
    @test all(==(0), zeros_like(r3))

    # putobs
    @test size(putobs(r1)) == (256, 256, 1)
    @test size(putobs(r2)) == (256, 256, 3, 1)
    @test size(putobs(r3)) == (256, 256, 3, 9, 1)

    # rmobs
    @test size(rmobs(putobs(r1))) == (256, 256)
    @test size(rmobs(putobs(r2))) == (256, 256, 3)
    @test size(rmobs(putobs(r3))) == (256, 256, 3, 9)
end

@testset "views" begin
    # Test Data
    x1 = collect(1:10)
    x2 = collect(21:30)
    v1 = ObsView(1:10, 1:10)
    v2 = ObsView(21:30, 1:10)

    # zipobs
    @test all(collect(zipobs(v1, v2)) .== collect(zip(x1, x2)))  # Test zipobs

    # repeatobs
    @test all(repeatobs(v1, 5) .== reduce(vcat, [x1 for _ in 1:5]))  # Test repeatobs
    @test all(repeatobs(zipobs(v1, v2), 2) .== reduce(vcat, [collect(zip(x1, x2)) for _ in 1:2]))  # zipobs + repeatobs

    # splitobs with shuffle
    @test first(splitobs(StableRNGs.StableRNG(123), 1:10, at=0.7)) == [4, 7, 2, 1, 3, 8, 5]
    @test last(splitobs(StableRNGs.StableRNG(123), 1:10, at=0.7)) == [6, 10, 9]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5])[1] == [4, 7, 2]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5])[2] == [1, 3, 8, 5, 6]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5])[3] == [10, 9]
    @test_throws ArgumentError splitobs(1:10, at=[0.3, 0.8])
    @test map(length, splitobs(1:10, at=[0.3, 0.7])) == [3, 7, 0]

    # splitobs without shuffle
    @test first(splitobs(StableRNGs.StableRNG(123), 1:10, at=0.7; shuffle=false)) == [1, 2, 3, 4, 5, 6, 7]
    @test last(splitobs(StableRNGs.StableRNG(123), 1:10, at=0.7; shuffle=false)) == [8, 9, 10]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5]; shuffle=false)[1] == [1, 2, 3]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5]; shuffle=false)[2] == [4, 5, 6, 7, 8]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5]; shuffle=false)[3] == [9, 10]

    # takeobs
    @test all(takeobs(v1, [2, 5, 8, 9]) .== x1[[2, 5, 8, 9]])  # takeobs
    @test_throws ArgumentError takeobs(v1, [0, 1, 2])

    # dropobs
    @test all(dropobs(v1, [1,2,3,5,6,8,9,10]) .== x1[[4,7]])  # dropobs

    # filterobs
    @test all(filterobs(iseven,  1:10) .== [2, 4, 6, 8, 10])
    @test all(filterobs(iseven,  v1) .== [2, 4, 6, 8, 10])

    # mapobs
    @test all(mapobs(x -> x * 2 + 1, v1) .== [3, 5, 7, 9, 11, 13, 15, 17, 19, 21])

    # sampleobs
    @test all(sampleobs(StableRNGs.StableRNG(126), v2, 4) .== [25, 24, 30, 26])
    @test length(sampleobs(v1, 0)) == 0
    @test_throws ArgumentError sampleobs(v2, -1)
    @test_throws ArgumentError sampleobs(v2, length(v2) + 1)

    # shuffleobs
    @test all(shuffleobs(StableRNGs.StableRNG(123), v1) .== [4, 7, 2, 1, 3, 8, 5, 6, 10, 9])

    # TileView
    tile = Raster(rand(rng, UInt16, 256, 256, 2, 8), (X, Y, Band, Ti))
    @test length(TileView(tile, 64)) == 16
    @test length(TileView(tile, 64, stride=32)) == 49
    @test map(size, TileView(tile, 64)[1:4:16]) == [(64, 64, 2, 8), (64, 64, 2, 8), (64, 64, 2, 8), (64, 64, 2, 8)]
    @test size(mapobs(tensor, TileView(tile, 64))[1:4:16]) == (64, 64, 2, 8, 4)
    @test typeof(mapobs(tensor, TileView(tile, 64))[1:4:16]) == Array{Float32, 5}
end

@testset "class metrics" begin
    # Accuracy - Prediction Rounding
    m = Metric(Accuracy())
    update!(m, [0.1, 0.8, 0.51, 0.49], [0, 1, 1, 0])
    @test compute(m) == 1

    # Accuracy - Multi Batch
    reset!(m)
    update!(m, [0.1, 0.6], [0, 1])
    update!(m, [0.7, 0.4], [1, 1])
    @test compute(m) == 0.75

    # Accuracy - Perfectly Incorrect
    reset!(m)
    update!(m, [0, 0, 0, 1], [1, 1, 1, 0])
    @test compute(m) == 0

    # Accuracy - Soft Labels
    reset!(m)
    update!(m, [0.1, 0.1, 0.1, 0.9], [0.95, 0.95, 0.95, 0.05])
    @test compute(m) == 0

    # Accuracy - One Hot Labels
    reset!(m)
    update!(m, hcat([0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.15, 0.85]), hcat([0, 1], [1, 0], [1, 0], [0, 1]))
    @test compute(m) == 0.75

    # MIoU - Prediction Rounding
    m = Metric(MIoU([0,1]))
    update!(m, [0.1, 0.8, 0.51, 0.49], [0, 1, 1, 0])
    @test compute(m) == 1

    # MIoU - Multi Batch
    reset!(m)
    update!(m, [0.1, 0.6], [0, 1])
    update!(m, [0.7, 0.4], [1, 1])
    @test compute(m) ≈ 0.5833333333333333

    # MIoU - Perfectly Incorrect
    reset!(m)
    update!(m, [0, 0, 0, 1], [1, 1, 1, 0])
    @test compute(m) ≈ 0 atol=1e-12

    # MIoU - Soft Labels
    reset!(m)
    update!(m, [0.1, 0.9, 0.7, 0.4], [0.05, 0.95, 0.95, 0.95])
    @test compute(m) ≈ 0.5833333333333333

    # MIoU - No Positive Labels
    reset!(m)
    update!(m, [0, 0, 0, 0], [0, 0, 0, 0])
    @test compute(m) == 1.0

    # Accuracy - One Hot Labels
    m = Metric(MIoU([1,2]))
    update!(m, hcat([0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.85, 0.15]), hcat([1, 0], [0, 1], [0, 1], [0, 1]))
    @test compute(m) ≈ 0.5833333333333333
end

@testset "tracker" begin
    # Initialize Tracker
    tracker = Tracker("train_acc" => Accuracy(), "test_acc" => Accuracy())

    # Update Metrics Matching Regex
    step!(tracker, r"train_", [0.1, 0.8], [0, 1])
    @test scores(tracker) == (epoch=1, train_acc=1.0, test_acc=0.0)

    # Update Metrics Matching Name
    step!(tracker, "test_acc", [0.6, 0.51], [1, 0])
    @test scores(tracker) == (epoch=1, train_acc=1.0, test_acc=0.5)

    # End Epoch
    epoch!(tracker)
    @test scores(tracker) == (epoch=2, train_acc=0.0, test_acc=0.0)

    # Update All Metrics
    step!(tracker, [0, 1, 1, 0], [0, 1, 1, 1])
    @test scores(tracker) == (epoch=2, train_acc=0.75, test_acc=0.75)

    # Find Best Epoch
    @test best_epoch(tracker, Max("test_acc")) == 1
    epoch!(tracker)
    @test best_epoch(tracker, Max("test_acc")) == 2
    @test best_epoch(tracker, Min("test_acc")) == 1

    # Test Score Printing
    @test printscores(tracker, epoch=1) == "epoch: 1  train_acc: 1.0  test_acc: 0.5"
end

@testset "models" begin
    # Test Data
    x1 = rand(rng, Float32, 128, 128, 4, 1)
    x2 = rand(rng, Float32, 128, 128, 2, 6, 1)
    encoders = [ResNet18(), ResNet34(), ResNet50(), ResNet101(), ResNet152()]

    # UNet
    for encoder in encoders
        unet = UNet(input=Single(channels=4), encoder=encoder)
        unet_ts = UNet(input=Series(channels=2), encoder=encoder)
        @test size(unet(x1)) == (128, 128, 1, 1)
        @test size(unet_ts(x2)) == (128, 128, 1, 1)
    end

    # Classifier
    for encoder in encoders
        classifier = Classifier(input=Single(channels=4), encoder=encoder, nclasses=10)
        @test size(classifier(x1)) == (10, 1)
    end

    # SSC_CNN
    ssc_cnn = SSC_CNN()
    lr = rand(Float32, 64, 64, 6, 1);
    hr = rand(Float32, 64, 64, 4, 1);
    @test size(ssc_cnn(lr, hr)) == (64, 64, 6, 1)
end