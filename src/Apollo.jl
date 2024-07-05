module Apollo

import Flux, MLUtils, Random
using Rasters, Match, Statistics
using Pipe: @pipe
using Match

const HasDims = Union{<:AbstractDimArray,<:AbstractDimStack}

include("utils.jl")
include("tensor.jl")
include("samplers.jl")
include("normalize.jl")
include("models/common.jl")
include("models/chain.jl")
include("models/unet.jl")
include("models/ssc_cnn.jl")
include("losses.jl")
include("metrics/common.jl")
include("metrics/classification.jl")
include("metrics/regression.jl")

export Tensor, TDim, W, H, C, N, L, TShape, WHCN, WHCLN, TDim, Width, Height, Channel, Length, Obs
export UNet, SSC_CNN, Chain
export TileSampler
export tensor, raster, catlayers, add_dim, apply
export binarycrossentropy, mae, mse
export AbstractMetric, ClassificationMetric, Metric, MIoU, Accuracy, Loss
export name, init, update, update!, compute, evaluate

end
