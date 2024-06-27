module Apollo

import Flux, MLUtils, Random
using Rasters, Match, Statistics
using Pipe: @pipe
using Match

const HasDims = Union{<:AbstractRaster,<:AbstractRasterStack,<:AbstractRasterSeries}

include("utils.jl")
include("samplers.jl")
include("models/common.jl")
include("models/chain.jl")
include("models/unet.jl")
include("models/ssc_cnn.jl")
include("losses.jl")

export UNet
export TileSampler
export tensor

end
