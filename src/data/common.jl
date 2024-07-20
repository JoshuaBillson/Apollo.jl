abstract type TileSource end

abstract type DType{N} end
struct Image{N} <: DType{N} end
struct Mask{N} <: DType{N} end
