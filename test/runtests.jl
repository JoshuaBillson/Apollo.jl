using Apollo
using Test
using Rasters
using Statistics
using ArchGDAL
using Random
using StableRNGs
import Flux
using Pipe: @pipe

const rng = StableRNG(123)

@testset "transforms" begin
    """
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

    # onehot and onecold
    logits = [1 2; 3 1]
    one_hot = reshape(cat([1 0; 0 1], [0 1; 0 0], [0 0; 1 0], dims=3), (2,2,3,1))
    @test all(Apollo.onecold(one_hot) .== [0 1; 2 0])

    # OneHot Transform
    t = OneHot(labels=[1,2,3])
    @test apply(t, Image(), logits, 123) == logits
    @test apply(t, WeightMask(), logits, 123) == logits
    @test all(apply(t, SegMask(), logits, 123) .== one_hot)


    # resample
    @test size(Apollo.resample(r1, 2.0, :bilinear)) == (512, 512, 3)
    @test size(Apollo.resample(r1, 0.5, :average)) == (128, 128, 3)
    @test size(Apollo.resample(r5, 2.0, :bilinear)) == (3, 256, 256, 9)
    @test size(Apollo.resample(r5, 0.5, :average)) == (3, 64, 64, 9)
    @test size(Apollo.resample(tensor(r5), 0.5, :average)) == (64, 64, 3, 9, 1)
    @test_throws ArgumentError Apollo.resample(r5, 0.5, :foo)
    @test_throws ArgumentError Apollo.resample(r5, 0, :average)
    @test_throws ArgumentError Apollo.resample(r5, -1, :average)

    # Resample transform
    vals = Set(r1.data)
    @test size(apply(Resample(2.0), Image(), r1, 123)) == (512, 512, 3)
    @test size(apply(Resample(1.5), Image(), r1, 123)) == (384, 384, 3)
    @test size(apply(Resample(0.5), Image(), r1, 123)) == (128, 128, 3)
    @test size(apply(Resample(0.5), Image(), r5, 123)) == (3, 64, 64, 9)
    @test size(apply(Resample(2.0), Image(), tensor(r5), 123)) == (256, 256, 3, 9, 1)
    @test all(x -> x in vals, apply(Resample(2), SegMask(), r1, 123))
    @test_throws ArgumentError Resample(0)

    # upsample
    @test size(Apollo.upsample(tensor(r1), 2, :bilinear)) == (512, 512, 3, 1) # bilinear
    @test size(Apollo.upsample(tensor(r1), 2, :nearest)) == (512, 512, 3, 1)  # nearest
    @test size(Apollo.upsample(tensor(r5), 2, :bilinear)) == (256, 256, 3, 9, 1)  # temporal dimension
    @test_throws ArgumentError Apollo.upsample(tensor(r5), 2, :foo)  # invalid method
    @test_throws AssertionError Apollo.upsample(tensor(r5), 0.5, :bilinear)  # invalid scale
    @test_throws ArgumentError Apollo.upsample(r5, 2, :bilinear)  # invalid type

    # resize
    @test size(Apollo.resize(r1, (512, 512), :bilinear)) == (512, 512, 3) # bilinear
    @test size(Apollo.resize(r1, (128, 128), :average)) == (128, 128, 3)  # average
    @test size(Apollo.resize(r1, (128, 512), :nearest)) == (128, 512, 3)  # difference scales per dimension
    @test size(Apollo.resize(r5, (256, 256), :bilinear)) == (3, 256, 256, 9)  # temporal dimension
    @test_throws ArchGDAL.GDAL.GDALError Apollo.resize(r5, (-256, 256), :bilinear)  # Invalid size
    @test_throws ArgumentError Apollo.resize(r5, (256, 256), :foo)  # invalid method

    # crop
    @test size(Apollo.crop(r1, 128)) == (128, 128, 3)  # crop at (1,1) (size)
    @test all(Apollo.crop(r1, 128) .== r1[X(1:128), Y(1:128)])  # crop at (1,1) (values)
    @test size(Apollo.crop(r1, 128, (129,129))) == (128, 128, 3)  # crop at (129,129) (size)
    @test all(Apollo.crop(r1, 128, (129,129)) .== r1[X(129:256), Y(129:256)])  # crop at (129,129) (values)
    @test size(Apollo.crop(r5, 64, (65,65))) == (3, 64, 64, 9)  # dims out of order (size)
    @test all(Apollo.crop(r5, 64, (65,65)) .== r5[X(65:128), Y(65:128)])  # dims out of order (values)
    @test size(Apollo.crop(tensor(r5), 64, (65,65))) == (64, 64, 3, 9, 1)  # tensor (size)
    @test all(Apollo.crop(tensor(r5), 64, (65,65)) .== tensor(r5)[65:128, 65:128, :, :, :])  # tensor (values)
    @test size(Apollo.crop(RasterStack(r5, layersfrom=Band), 64, (65,65))) == (64, 64, 9)  # stack (size)
    @test all(Apollo.crop(RasterStack(r5, layersfrom=Band), 64, (65,65))[:Band_1] .== r5[X(65:128), Y(65:128), Band(1)])  # stack (values)
    @test_throws ArgumentError Apollo.crop(r5, 64, (66,65))  # Out of Bounds (raster)
    @test_throws ArgumentError Apollo.crop(tensor(r5), 64, (66,65))  # Out of Bounds (tensor)
    @test_throws ArgumentError Apollo.crop(r5, 129)  # tile too large (raster)
    @test_throws ArgumentError Apollo.crop(tensor(r5), 129)  # tile too large (tensor)
    @test_throws ArgumentError Apollo.crop(r5, 0)  # zero tilesize
    @test_throws ArgumentError Apollo.crop(r5, -1)  # negative tilesize
    @test_throws MethodError Apollo.crop(r5, 2.5)  # float tilesize
    @test_throws ArgumentError Apollo.crop(r5, 128, (-1, -1))  # negative ul

    # Crop transform
    @test size(apply(Crop(128), Image(), r1, 123)) == (128, 128, 3)
    @test size(apply(Crop(32), SegMask(), r5, 123)) == (3, 32, 32, 9)
    @test size(apply(Crop(32), SegMask(), tensor(r5), 123)) == (32, 32, 3, 9, 1)

    # Test Random Outcome
    @test isapprox(sum([Apollo._apply_random(rand(1:1000000), 0.3) for _ in 1:10000]) / 10000, 0.3, atol=0.02)

    # RandomCrop transform
    @test size(apply(RandomCrop(128), Image(), r1, 123)) == (128, 128, 3)
    @test size(apply(RandomCrop(32), SegMask(), r5, 123)) == (3, 32, 32, 9)
    @test size(apply(RandomCrop(32), Image(), tensor(r5), 123)) == (32, 32, 3, 9, 1)
    @test apply(RandomCrop(32), Image(), r1, 123) != Apollo.crop(r1, 32)

    # flipX
    @test catlayers(Apollo.flipX(RasterStack(r1, layersfrom=Band)), Band) == Apollo.flipX(r1)  # stack
    @test Apollo.flipX(tensor(r1)) == tensor(Apollo.flipX(r1))  # tensor
    @test tensor(apply(FlipX(1), Image(), r1, 123)) == apply(FlipX(1), Image(), tensor(r1), 123)
    @test size(apply(FlipX(0), Image(), r1, 123)) == (256, 256, 3)
    @test size(apply(FlipX(0), Image(), r5, 123)) == (3, 128, 128, 9)
    @test apply(FlipX(1), Image(), r1, 123) == flipX(r1)
    @test apply(FlipX(0), Image(), r1, 123) == r1
    @test_throws ArgumentError FlipX(1.2)
    @test_throws ArgumentError FlipX(-1)

    # flipY
    @test catlayers(Apollo.flipY(RasterStack(r1, layersfrom=Band)), Band) == Apollo.flipY(r1)  # stack
    @test Apollo.flipY(tensor(r1)) == tensor(Apollo.flipY(r1))  # tensor
    @test tensor(apply(FlipY(1), Image(), r1, 123)) == apply(FlipY(1), Image(), tensor(r1), 123)
    @test size(apply(FlipY(0), Image(), r1, 123)) == (256, 256, 3)
    @test size(apply(FlipY(0), Image(), r5, 123)) == (3, 128, 128, 9)
    @test apply(FlipY(1), Image(), r1, 123) == flipY(r1)
    @test apply(FlipY(0), Image(), r1, 123) == r1
    @test_throws ArgumentError FlipY(1.2)
    @test_throws ArgumentError FlipY(-1)

    # rot90
    @test catlayers(Apollo.rot90(RasterStack(r1, layersfrom=Band)), Band) == Apollo.rot90(r1)  # stack
    @test Apollo.rot90(tensor(r1)) == tensor(Apollo.rot90(r1)) # tensor
    @test tensor(apply(Rot90(1), Image(), r1, 123)) == apply(Rot90(1), Image(), tensor(r1), 123)
    @test size(apply(Rot90(0), Image(), r1, 123)) == (256, 256, 3)
    @test size(apply(Rot90(0), Image(), r5, 123)) == (3, 128, 128, 9)
    @test apply(Rot90(1), Image(), r1, 123) == rot90(r1)
    @test apply(Rot90(0), Image(), r1, 123) == r1
    @test_throws ArgumentError Rot90(1.2)
    @test_throws ArgumentError Rot90(-1)


    # ComposedTransform
    t = Tensor() |> Crop(128)
    transformed = apply(t, Image(), r1, 123)
    @test size(transformed) == (128, 128, 3, 1)
    @test transformed isa Array{Float32,4}
    """
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

@testset "TileView" begin
    tile = Raster(rand(rng, UInt16, 256, 256, 2, 8), (X, Y, Band, Ti))
    @test length(TileView(tile, 64)) == 16
    @test length(TileView(tile, 64, stride=32)) == 49
    @test map(size, TileView(tile, 64)[1:4:16]) == [(64, 64, 2, 8), (64, 64, 2, 8), (64, 64, 2, 8), (64, 64, 2, 8)]
end

@testset "class losses" begin
    # Initialize Data
    y = rand([0.0f0, 1.0f0], 3, 3, 1, 1)
    ŷ = rand(Float32, 3, 3, 1, 1)
    y_hot = Apollo.onehot(y, 0:1)
    ŷ_hot = cat(1 .- ŷ, ŷ, dims=3)

    # Cross Entropy
    ce = Apollo.CrossEntropy()
    bce = Apollo.BinaryCrossEntropy()
    @test bce(ŷ, y) ≈ Flux.Losses.binarycrossentropy(ŷ, y)
    @test bce(ŷ_hot[:,:,2:2,:], y_hot[:,:,2:2,:]) ≈ bce(ŷ, y)
    @test isapprox(bce(y, y), 0, atol=1e-5)
end

@testset "models" begin
    # Test Data
    x1 = rand(rng, Float32, 128, 128, 4, 1)
    x2 = rand(rng, Float32, 128, 128, 2, 6, 1)
    encoders = [
        ResNet(depth=18, weights=:Nothing), 
        ResNet(depth=34, weights=:Nothing), 
        ResNet(depth=50, weights=:Nothing), 
        ResNet(depth=101, weights=:Nothing), 
        ResNet(depth=152, weights=:Nothing) ]

    # UNet
    for encoder in encoders
        unet = UNet(encoder=encoder, inchannels=4)
        @test size(unet(x1)) == (128, 128, 1, 1)
    end

    # DeeplabV3+
    for encoder in encoders
        deeplab = DeeplabV3(encoder=encoder, inchannels=4)
        @test size(deeplab(x1)) == (128, 128, 1, 1)
    end

    # SSC_CNN
    ssc_cnn = SSC_CNN()
    lr = rand(Float32, 64, 64, 6, 1);
    hr = rand(Float32, 64, 64, 4, 1);
    @test size(ssc_cnn(lr, hr)) == (64, 64, 6, 1)
end