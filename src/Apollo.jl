module Apollo

import Flux, MLUtils, Random
using Rasters, Match, Statistics
using Pipe: @pipe
using Match

const HasDims = Union{<:AbstractDimArray,<:AbstractDimStack}

include("utils.jl")

include("data/tensor.jl")
include("data/samplers.jl")
include("data/pipeline.jl")
include("data/normalize.jl")

include("models/common.jl")
include("models/chain.jl")
include("models/unet.jl")
include("models/ssc_cnn.jl")
include("models/r2unet.jl")

include("losses.jl")

include("metrics/common.jl")
include("metrics/classification.jl")
include("metrics/regression.jl")

# Utils
export catlayers, add_dim, folddims, foldlayers, apply

# Data
export Tensor, TDim, W, H, C, N, L, TShape, WHCN, WHCLN, TDim, Width, Height, Channel, Length, Obs
export TileSampler
export tensor, raster

# Losses
export binarycrossentropy, mae, mse

# Metrics
export AbstractMetric, ClassificationMetric, Metric, MIoU, Accuracy, Loss
export name, init, update, update!, compute, evaluate

# Models
export UNet, SSC_CNN, Chain

end
