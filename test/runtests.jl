using Apollo
using Test
using Rasters

@testset "Apollo.jl" begin
    # Test Tensor
    x = Raster(rand(Float32, 256, 256, 3), (X, Y, Band))
    y = Raster(rand(Float32, 3, 256, 256), (Band, X, Y))
    z = Raster(rand(Float32, 128, 128, 3), (X, Y, Band))
    @test size(tensor(x, dims=(X,Y,Band))) == (256, 256, 3, 1)  # simple case
    @test size(tensor(x, dims=(Band,Y,X))) == (3, 256, 256, 1)  # re-order dims
    @test size(tensor(x, y, dims=(X,Y,Band))) == (256, 256, 3, 2)  # Mutiple rasters with different dim order
    @test size(tensor(x, y, dims=(X,Band,Y))) == (256, 3, 256, 2)  # Mutiple rasters with different dim order
    @test_throws AssertionError tensor(x, dims=(X,Y,Band,X))  # Test Duplicate Dims
    @test_throws AssertionError tensor(x, dims=(X,Y))  # Test Missing Dims
    @test_throws AssertionError tensor(x, dims=(Y,X,Band,Ti))  
    @test_throws AssertionError tensor(x, dims=(Y,X,Ti,Band))
    @test_throws AssertionError tensor(x, dims=(X,Y,X))  # Test Duplicate with correct number of dims
    @test_throws AssertionError tensor(x, dims=(X,Y,Z,Ti))  # Test missing with correct number of dims
    @test_throws AssertionError tensor(x, y, z, dims=(X,Y,Band))  # Rasters with different sizes
end
