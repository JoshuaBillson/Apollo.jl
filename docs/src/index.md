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
AbstractIterator
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
FlipX
FlipY
FilteredTransform
Normalize
RandomCrop
Resample
Rot90
Tensor

apply
crop
denormalize
flipX
flipY
normalize
raster
resample
resize
rot90
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

# Losses

```@docs
AbstractLoss
MaskedLoss
WeightedLoss
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