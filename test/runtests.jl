using Apollo
using Test
using Rasters

@testset "Apollo.jl" begin
    # Test Tensor
    x = Raster(rand(Float32, 256, 256, 3), (X, Y, Band))
    y = Raster(rand(Float32, 3, 256, 256), (Band, X, Y))
    z = Raster(rand(Float32, 128, 128, 3), (X, Y, Band))
    t1 = Raster(rand(Float32, 128, 128, 3, 9), (X, Y, Band, Ti))
    t2 = Raster(rand(Float32, 3, 128, 128, 9), (Band, X, Y, Ti))
    b = Raster(rand(Float32, 256, 256), (X, Y))
    @test size(tensor(x)) == (256, 256, 3, 1)  # simple case
    @test size(tensor(x, y)) == (256, 256, 3, 2)  # Mutiple rasters with different dim order
    @test size(tensor(t1)) == (128, 128, 3, 9, 1) 
    @test size(tensor(t2)) == (128, 128, 3, 9, 1)
    @test size(tensor(t1, t2)) == (128, 128, 3, 9, 2)
    @test size(tensor(b, b)) == (256, 256, 1, 2)  # Missing Bands
    @test_throws DimensionMismatch tensor(x, y, t1)  # Extra Dimension
    @test_throws DimensionMismatch tensor(t1, t2, x)  # Missing Dimension
    @test_throws DimensionMismatch tensor(x, y, b)  # Mismatched Sizes
    @test_throws DimensionMismatch tensor(x, y, z)  # Mismatched Sizes

    # Test Utilities
    x = Raster(rand(Float32, 128, 128), (X,Y))
    @test size(putdim(x, Band)) == (128,128,1)
    @test hasdim(putdim(x, Band), Band)
    @test size(putdim(x, Ti)) == (128,128,1)
    @test hasdim(putdim(x, Ti), Ti)
    @test putdim(x, X) == x
    @test size(putdim(x, (Z, Band, Ti, X, Y))) == (128,128,1,1,1)
    @test hasdim(putdim(x, (Z, Band, Ti, X, Y)), Z)
    @test hasdim(putdim(x, (Z, Band, Ti, X, Y)), Band)
    @test hasdim(putdim(x, (Z, Band, Ti, X, Y)), Ti)

    # Test Views
    x1 = collect(1:10)
    x2 = collect(21:30)
    v1 = ObsView(1:10, 1:10)
    v2 = ObsView(21:30, 1:10)
    @test all(collect(zipobs(v1, v2)) .== collect(zip(x1, x2)))  # Test zipobs
    @test all(repeatobs(v1, 5) .== reduce(vcat, [x1 for _ in 1:5]))  # Test repeatobs
    @test all(repeatobs(zipobs(v1, v2), 2) .== reduce(vcat, [collect(zip(x1, x2)) for _ in 1:2]))  # zipobs + repeatobs
    @test all(takeobs(v1, [2, 5, 8, 9]) .== x1[[2, 5, 8, 9]])  # takeobs
    @test all(dropobs(v1, [1,2,3,5,6,8,9,10]) .== x1[[4,7]])  # dropobs
end
