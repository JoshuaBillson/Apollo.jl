module Apollo

import Flux, Metalhead, ImageCore, Random, Tables, PrettyTables, MosaicViews
using Rasters, Match, Statistics
using Pipe: @pipe
using Accessors: @set
using DataStructures: OrderedDict

const HasDims = Union{<:AbstractRaster,<:AbstractRasterStack}

# Utilities
include("utils.jl")

# Visualization
include("viz.jl")

# Samplers
include("data/common.jl")
include("data/samplers.jl")

# Views
include("data/views.jl")

# Transforms
include("data/methods.jl")
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

# Metrics
include("metrics/common.jl")
include("metrics/classification.jl")
include("metrics/regression.jl")
include("metrics/tracker.jl")

# Training
include("training/losses.jl")
include("training/tasks.jl")
include("training/training.jl")

# Utils
export catlayers, putdim, folddims, foldlayers, putobs, rmobs, vec2array, ones_like, zeros_like, unzip, stackobs, todevice

# Visualization
export linear_stretch, rgb, binmask, mosaicview

# Transforms
export DType, Image, Mask, AbstractTransform
export Tensor, Normalize, DeNormalize, Resample, Crop, RandomCrop, FilteredTransform, ComposedTransform
export tensor, raster, transform, apply, normalize, denormalize, resample, upsample, resize, crop

# Samplers
export TileSource, TileSampler, TileSeq

# Views
export AbstractView, MappedView, JoinedView, ObsView
export splitobs, zipobs, repeatobs, takeobs, dropobs, filterobs, mapobs, sampleobs, shuffleobs

# Losses
export AbstractLoss, BinaryCrossEntropy, MeanAbsoluteError, MeanSquaredError, DiceLoss, MixedLoss

# Metrics
export AbstractMetric, ClassificationMetric, RegressionMetric
export MIoU, Accuracy, Loss, MSE
export name, init, update, update!, reset!, compute, evaluate

# Tracker
export Order, Max, Min, Tracker
export metrics, step!, epoch!, best_epoch, current_epoch, scores, printscores

# Layers
export SeparableConv, ConvBlock, Conv, LSTM

# Encoders
export AbstractEncoder, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, StandardEncoder

# Backbones
export Classifier, UNet, R2UNet, DeeplabV3, SSC_CNN

# Tasks
export BinarySegmentation

end
