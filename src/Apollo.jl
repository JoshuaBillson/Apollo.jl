module Apollo

import Flux, Metalhead, ImageCore, Random, Tables, PrettyTables, MosaicViews
using Rasters, Match, Statistics, ArchGDAL
using Pipe: @pipe
using Accessors: @set
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
export AbstractIterator, TileView, MappedView, JoinedView, ObsView, ZippedView, TransformedView
export splitobs, zipobs, repeatobs, takeobs, dropobs, filterobs, mapobs, sampleobs, shuffleobs

# Transforms
include("data/methods.jl")
include("data/transforms.jl")
export DType, AbstractImage, AbstractMask, Image, SegMask, WeightMask, AbstractTransform
export Tensor, OneHot, Normalize, DeNormalize, Resample, Crop, RandomCrop, FlipX, FlipY, Rot90, FilteredTransform, ComposedTransform
export tensor, raster, onehot, onecold, transform, apply, normalize, denormalize, resample, upsample, resize, crop, flipX, flipY, rot90

# Metrics
include("metrics/interface.jl")
include("metrics/classification.jl")
include("metrics/regression.jl")
export AbstractMetric, ClassificationMetric, RegressionMetric, Metric
export MIoU, Accuracy, Loss, MSE
export name, init, update, update!, reset!, compute

# Metric Logging
include("metrics/tracker.jl")
export Order, Max, Min, Tracker, MetricLogger
export step!, epoch!, best_epoch, current_epoch, scores, printscores

# Model Evaluation
include("metrics/evaluation.jl")
export evaluate, evaluate!

# Training
include("training/losses.jl")
include("training/training.jl")
export AbstractLoss, BinaryCrossEntropy, MeanAbsoluteError, MeanSquaredError, DiceLoss, MixedLoss, MaskedLoss
export WeightedLoss, WeightedBinaryCrossEntropy, WeightedMeanAbsoluteError, WeightedMeanSquaredError
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
export AbstractEncoder, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, StandardEncoder
export build_encoder

# Models
include("models/classifiers.jl")
include("models/unet.jl")
include("models/ssc_cnn.jl")
export Classifier, SegmentationModel, SSC_CNN

end