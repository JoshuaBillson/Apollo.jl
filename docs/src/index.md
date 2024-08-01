```@meta
CurrentModule = Apollo
```

# Apollo

Documentation for [Apollo](https://github.com/JoshuaBillson/Apollo.jl).

# Utility Methods

```@docs
binmask
catlayers
folddims
foldlayers
linear_stretch
mosaicview
ones_like
putdim
putobs
rgb
rmobs
stackobs
todevice
unzip
vec2array
zeros_like
```

# Data Views
```@docs
AbstractView
JoinedView
MappedView
ObsView
TileSampler
TileSeq

dropobs
filterobs
mapobs
repeatobs
sampleobs
shuffleobs
splitobs
takeobs
zipobs
```

# Transforms
```@docs
DType
Image
Mask

AbstractTransform
RandomTransform
ComposedTransform
Crop
DeNormalize
FilteredTransform
Normalize
RandomCrop
Resample
Tensor

apply
crop
denormalize
normalize
raster
resample
resize
tensor
transform
upsample
```

# Metrics

```@autodocs
Modules = [Apollo]
Pages = [
    "src/metrics/common.jl", 
    "src/metrics/tracker.jl", 
    "src/metrics/classification.jl", 
    "src/metrics/regression.jl", 
]
```

# Layers

```@docs
ConvBlock
LSTM
SeparableConv
```

# Encoders

```@docs
AbstractEncoder
ResNet18
ResNet34
ResNet50
ResNet101
ResNet152
StandardEncoder
```

# Models

```@docs
Classifier
DeeplabV3
R2UNet
SSC_CNN
UNet
```