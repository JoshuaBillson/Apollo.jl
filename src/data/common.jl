abstract type TileSource end

"""
The super-type of all data types.
"""
abstract type DType{N} end

"""
    Image([name])

Represents a type consisting of image data.
"""
struct Image{N} <: DType{N} end

Image() = Image(:x)
Image(x::Symbol) = Image{x}()

"""
    Mask([name])

Represents an instance of mask data.
"""
struct Mask{N} <: DType{N} end

Mask() = Mask(:y)
Mask(x::Symbol) = Mask{x}()