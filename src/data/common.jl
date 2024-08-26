abstract type TileSource end

"""
The super-type of all data types.
"""
abstract type DType{N} end

abstract type AbstractImage{N} <: DType{N} end

abstract type AbstractMask{N} <: DType{N} end

"""
    Image([name])

Represents a type consisting of image data.
"""
struct Image{N} <: AbstractImage{N} end

Image() = Image(:x)
Image(x::Symbol) = Image{x}()

Base.show(io::IO, x::Image{N}) where N = print(io, "Image(:$N)")

"""
    SegMask([name])

Represents an instance of a segmentation mask.
"""
struct SegMask{N} <: AbstractMask{N} end

SegMask() = SegMask(:y)
SegMask(x::Symbol) = SegMask{x}()

"""
    WeightMask([name])

Represents an instance of a weight mask.
"""
struct WeightMask{N} <: AbstractMask{N} end

WeightMask() = WeightMask(:w)
WeightMask(x::Symbol) = WeightMask{x}()