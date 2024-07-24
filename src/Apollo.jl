module Apollo

import Flux, Metalhead, ImageCore, Random
using Rasters, Match, Statistics
using Pipe: @pipe
using Match

const HasDims = Union{<:AbstractRaster,<:AbstractRasterStack}

# Utilities
include("utils.jl")

# Samplers
include("data/common.jl")
include("data/samplers.jl")

# Views
include("data/views.jl")

# Transforms
include("data/methods.jl")
include("data/tensor.jl")
include("data/normalize.jl")
include("data/transforms.jl")

# Models
include("models/layers.jl")
include("models/encoders.jl")
include("models/classifiers.jl")
include("models/unet.jl")
include("models/deeplab.jl")
include("models/ssc_cnn.jl")
include("models/r2unet.jl")

# Losses
include("losses.jl")

# Metrics
include("metrics/common.jl")
include("metrics/classification.jl")
include("metrics/regression.jl")

# Utils
export catlayers, putdim, folddims, foldlayers, putobs, dropobs, vec2array, ones_like, zeros_like

# Data

# Transforms
export DType, Image, Mask, AbstractTransform
export Tensor, Normalize, DeNormalize, Resample, Crop, RandomCrop, FilteredTransform, ComposedTransform
export transform, apply, tensor, normalize, denormalize, resample, upsample, resize, crop

# Samplers
export TileSource, TileSampler, TileSeq

# Views
export MappedView, JoinedView, ObsView
export splitobs, zipobs, repeatobs, takeobs, dropobs, filterobs, mapobs, sampleobs, shuffleobs

# Losses
export binarycrossentropy, mae, mse

# Metrics
export AbstractMetric, ClassificationMetric, Metric, MIoU, Accuracy, Loss
export name, init, update, update!, compute, evaluate

# Layers
export SeparableConv, ConvBlock, Conv

# Encoders
export AbstractEncoder, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, StandardEncoder

# Models
export Classifier, UNet, R2UNet, DeeplabV3, SSC_CNN

end
