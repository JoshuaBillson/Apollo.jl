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
    @test size(tensor(WHCN, x)) == (256, 256, 3, 1)  # simple case
    @test size(tensor(WHCN, x, y)) == (256, 256, 3, 2)  # Mutiple rasters with different dim order
    @test size(tensor(WHCLN, t1)) == (128, 128, 3, 9, 1) 
    @test size(tensor(WHCLN, t2)) == (128, 128, 3, 9, 1)
    @test size(tensor(WHCLN, t1, t2)) == (128, 128, 3, 9, 2)
    @test size(tensor(WHCN, b, b)) == (256, 256, 1, 2)  # Missing Bands
    @test_throws ArgumentError tensor(WHCN, t1)  # Extra Dimension
    @test_throws ArgumentError tensor(WHCN, x, y, t1)  # Extra Dimension
    @test_throws ArgumentError tensor(WHCLN, x)  # Missing Dimension
    @test_throws ArgumentError tensor(WHCLN, t1, t2, x)  # Missing Dimension
    @test_throws DimensionMismatch tensor(WHCN, x, y, b)  # Mismatched Sizes
    @test_throws DimensionMismatch tensor(WHCN, x, y, z)  # Mismatched Sizes
end
