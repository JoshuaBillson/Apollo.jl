module Apollo

import Flux, Metalhead, ImageCore, Random, Tables, PrettyTables, MosaicViews
using Rasters, Match, Statistics, ArchGDAL
using Pipe: @pipe
using Accessors: @set
using ArgCheck: @argcheck
using DataStructures: OrderedDict

const HasDims = Union{<:AbstractRaster,<:AbstractRasterStack}

# Utilities
include("utils.jl")
export catlayers, putdim, folddims, foldlayers, putobs, rmobs, vec2array, ones_like, zeros_like, unzip, stackobs, todevice

# Visualization
include("viz.jl")
export linear_stretch, rgb, binmask, mosaicview

# Views
include("data/common.jl")
include("data/views.jl")
export TileView

# Transforms
include("data/methods.jl")
include("data/transforms.jl")
export DType, AbstractImage, AbstractMask, Image, SegMask, WeightMask, AbstractTransform
export Tensor, OneHot, Resample, Crop, RandomCrop, FlipX, FlipY, Rot90, FilteredTransform, ComposedTransform
export tensor, raster, onehot, onecold, transform, apply, resample, upsample, resize, crop, flipX, flipY, rot90

# Training
include("training/losses.jl")
include("training/training.jl")
export AbstractLoss, BinaryCrossEntropy, MeanAbsoluteError, MeanSquaredError, BinaryDice, MultiClassDice, MixedLoss, MaskedLoss
export WeightedLoss, WeightedBinaryCrossEntropy
export update!

# Layers
include("models/layers.jl")
export SeparableConv, ConvBlock, Conv, LSTM

# Inputs
include("models/input.jl")
export RasterInput, SeriesInput
export build_input

# Encoders
include("models/encoders.jl")
export AbstractEncoder, ResNet, StandardEncoder
export build_encoder

# Models
include("models/classifiers.jl")
include("models/unet.jl")
include("models/deeplab.jl")
include("models/ssc_cnn.jl")
include("models/vitae.jl")
include("models/swin.jl")
include("models/pvt.jl")
export Classifier, UNet, DeeplabV3, SSC_CNN

end