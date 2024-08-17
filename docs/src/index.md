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
TileView
TransformedView
ZippedView

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

```@docs
AbstractMetric
ClassificationMetric
RegressionMetric
Metric
Accuracy
MIoU
MSE
Loss
compute
evaluate
init
name
reset!
update
update!
```

# Performance Tracking
```@docs
Tracker
MetricLogger
Max
Min

best_epoch
current_epoch
epoch!
printscores
scores
step!
```

# Layers

```@docs
ConvBlock
LSTM
SeparableConv
```

# Inputs
```@docs
AbstractInput
Single
Series
build_input
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
build_encoder
```

# Models

```@docs
Classifier
SSC_CNN
UNet
```

# Index

```@index
```