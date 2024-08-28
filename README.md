# Apollo

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoshuaBillson.github.io/Apollo.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaBillson.github.io/Apollo.jl/dev/)
[![Build Status](https://github.com/JoshuaBillson/Apollo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaBillson/Apollo.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JoshuaBillson/Apollo.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JoshuaBillson/Apollo.jl)

Our first step is to load our dataset with `Rasters.jl`.
```julia
labels = Rasters.Raster("data/RiceSC_South_S2/labels.tif")
features = Rasters.Raster("data/RiceSC_South_S2/2018-01-05_10m.tif")
```

Next, we compute the mean and standard so that we can normalize our features.
```julia
μ = Apollo.folddims(mean, features, dims=Rasters.Band)
σ = Apollo.folddims(std, features, dims=Rasters.Band)
```

Satellite images are often far too large to work with as a whole.
To solve this problem, we wrap our data inside of a `TileView` object to cut each raster into
512x512 non-overlapping tiles. `TileView` also accepts an optional `stride` keyword to generate
overlapping tiles.
```julia
xsampler = A.TileView(features, 512)
ysampler = A.TileView(labels, 512)
data = A.zipobs(xsampler, ysampler)


`TileView` produces a lazy iterator over the generated tiles, which we can then split
into a 70/30 train/test split with the `splitobs` method.
```julia
train, test = A.splitobs(data, at=0.7)
```

We now define our data pipeline as a sequence of transformations.
We first want to convert our rasters into tensors and normalize our features according
to the previously computed mean and standard deviations. We can then apply a series
of random transformations to augment our training data.
```julia
# Define Data Transforms
dtypes = (A.Image(), A.SegMask())
preprocess = A.Tensor() |> A.Normalize(μ, σ)
aug = A.RandomCrop((256,256)) |> A.FlipX(0.5) |> A.FlipY(0.5) |> A.Rot90(0.25)

# Apply Transforms
train_pipe = Apollo.transform(preprocess |> aug, dtypes, train)
test_pipe = Apollo.transform(preprocess, dtypes, test)
```

Now that our images have been cut into tiles, split into train and test, and placed into a pipeline, we can
now create a pair of `Flux.DataLoader` objects to shuffle and split our data into batches.
```julia
traindata = Flux.DataLoader(train_pipe, batchsize=8, shuffle=true)
testdata = Flux.DataLoader(test_pipe, batchsize=4)
```


To build our model, we'll first need to specify the types of both the input and the encoder. 
In this example, our input is simply a single raster with 4 bandsand our encoder is ResNet18 network with ImageNet pre-trained weights.
```julia
input = A.RasterInput(channels=4)
encoder = A.ResNet18(weights=:ImageNet)
model = A.SegmentationModel(input=input, encoder=encoder, nclasses=1) |> Flux.gpu
```

Now we need to define and initialize our optimizer. 
We'll also freeze the weights in the encoder to avoid losing the pre-trained weights.
```julia
opt = Flux.Optimisers.Adam(5e-4)
opt_state = Flux.Optimisers.setup(opt, model)
Flux.Optimisers.freeze!(opt_state.encoder)
```

`Trackers` are used to keep track of one or more metrics over the course of a training run.
We'll initialize one here to log our loss on both the training and test data.
```julia
tracker = A.Tracker(
"train_loss" => A.Loss(loss), 
"test_loss" => A.Loss(loss), 
)
```

Now we're ready to train our model.
We run our training loop for 30 epochs, where each epoch consists of a training and evaluation step.
We call `train!` to fit our model on a single pass over the provided dataset.
We then update our training and test loss by calling `evaluate!`.
Finally, we log our current loss and tell our tracker to terminate the current epoch.
```julia
for epoch in 1:30

  # Update Model
  A.train!(loss, model, opt_state, Flux.gpu(traindata))

  # Evaluate Train and Test Performance
  A.evaluate!(model, traindata, tracker, gpu=true, metrics=r"train_")
  A.evaluate!(model, testdata, tracker, gpu=true, metrics=r"test_")

  # End Epoch
  @info A.printscores(tracker)
  A.epoch!(tracker)
end
```

Finally, we're ready to evaluate our trained model in terms of its loss, accuracy, and mIoU.
```julia
metrics = [A.Loss(loss), A.MIoU([0,1]), A.Accuracy()]
train_metrics = A.evaluate(model, traindata, metrics..., gpu=true)
test_metrics = A.evaluate(model, testdata, metrics..., gpu=true)
```
