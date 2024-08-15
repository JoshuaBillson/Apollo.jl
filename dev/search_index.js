var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Apollo","category":"page"},{"location":"#Apollo","page":"Home","title":"Apollo","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Apollo.","category":"page"},{"location":"#Utility-Methods","page":"Home","title":"Utility Methods","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"binmask\ncatlayers\nfolddims\nfoldlayers\nlinear_stretch\nmosaicview\nones_like\nputdim\nputobs\nrgb\nrmobs\nstackobs\ntodevice\nunzip\nvec2array\nzeros_like","category":"page"},{"location":"#Apollo.binmask","page":"Home","title":"Apollo.binmask","text":"binmask(x)\n\nVisualze x as a black/white binary mask.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.catlayers","page":"Home","title":"Apollo.catlayers","text":"catlayers(x::AbstractRasterStack, dim)\n\nConcatenate the layers of x along the dimension given by dim.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.folddims","page":"Home","title":"Apollo.folddims","text":"folddims(f, xs::AbstractRaster; dims=Band)\n\nApply the reducer function f to all non-missing elements in each slice of x WRT dims.\n\nArguments\n\nf: A function that reduces an array of values to a singular value.\nx: An AbstractRaster over which we want to fold.\ndims: The dimension used to generate each slice that is passed to f.\n\nExample\n\njulia> μ = folddims(mean, raster, dims=Band)\n6-element Vector{Float32}:\n 0.09044644\n 0.23737456\n 0.30892986\n 0.33931717\n 0.16186203\n 0.076255515\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.foldlayers","page":"Home","title":"Apollo.foldlayers","text":"foldlayers(f, x::AbstractRasterStack)\n\nApply the reducer f to all non-missing elements in each layer of x.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.linear_stretch","page":"Home","title":"Apollo.linear_stretch","text":"linear_stretch(x::AbstractArray{<:Real,3}, lb::Vector{<:Real}, ub::Vector{<:Real})\n\nPerform a linear histogram stretch on x such that lb is mapped to 0 and ub is mapped to 1. Values outside the interval [lb, ub] will be clamped.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.mosaicview","page":"Home","title":"Apollo.mosaicview","text":"mosaicview(size::Tuple{Int,Int}, imgs...)\n\nPlot a mosaic of images with size (rows, cols).\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.ones_like","page":"Home","title":"Apollo.ones_like","text":"ones_like(x::AbstractArray)\n\nConstruct an array of ones with the same size and element type as x.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.putdim","page":"Home","title":"Apollo.putdim","text":"putdim(raster::AbstractRaster, dims)\n\nAdd the provided singleton dim(s) to raster. Does nothing if dims is already present.\n\nExample\n\njulia> r = Raster(rand(512,512), (X,Y));\n\njulia> putdim(r, Band)\n╭─────────────────────────────╮\n│ 512×512×1 Raster{Float64,3} │\n├─────────────────────────────┴─────────────────────────────── dims ┐\n  ↓ X   ,\n  → Y   ,\n  ↗ Band Sampled{Int64} Base.OneTo(1) ForwardOrdered Regular Points\n├─────────────────────────────────────────────────────────── raster ┤\n  extent: Extent(X = (1, 512), Y = (1, 512), Band = (1, 1))\n\n└───────────────────────────────────────────────────────────────────┘\n[:, :, 1]\n 0.107662  0.263251    0.786834  0.334663  …  0.316804   0.709557    0.478199\n 0.379863  0.532268    0.635206  0.33514      0.402433   0.413602    0.657538\n 0.129775  0.283808    0.327946  0.727027     0.685844   0.847777    0.435326\n 0.73348   0.00705636  0.178885  0.381932     0.146575   0.310242    0.159852\n ⋮                                         ⋱                         ⋮\n 0.330857  0.52704     0.888379  0.811084  …  0.0660687  0.00230472  0.448761\n 0.698654  0.510846    0.916446  0.621061     0.23648    0.510697    0.113338\n 0.600629  0.116626    0.567983  0.174267     0.089853   0.443758    0.667935\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.putobs","page":"Home","title":"Apollo.putobs","text":"putobs(x::AbstractArray)\n\nAdd an N+1 obervation dimension of size 1 to the tensor x.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.rgb","page":"Home","title":"Apollo.rgb","text":"rgb(x, lb, ub; bands=[1,2,3])\n\nVisualze the specified bands of x as an rgb image.\n\nParameters\n\nx: The AbstractArray to be visualized. Must contain at least 3 bands.\nlb: A vector of lower bounds for each channel in x. Used by linear_stretch.\nub: A vector of upper bounds for each channel in x. Used by linear_stretch.\nbands: A vector of band indices to assign to red, green, and blue, respectively.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.rmobs","page":"Home","title":"Apollo.rmobs","text":"rmobs(x::AbstractArray)\n\nRemove the observation dimension from the tensor x.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.stackobs","page":"Home","title":"Apollo.stackobs","text":"stackobs(x...)\nstackobs(x::AbstractVector)\n\nStack the elements in x as if they were observations in a batch. If x is an AbstractArray,  elements will be concatenated along the Nth dimension. Other data types will simply be placed in a Vector in the same order as they were received. Special attention is paid to a Vector of Tuples, where each tuple represents a single observation, such as a feature/label pair. In this case, the tuples will first be unzipped, before their constituent elements are then stacked as usual.\n\nExample\n\njulia> stackobs(1, 2, 3, 4, 5)\n5-element Vector{Int64}:\n 1\n 2\n 3\n 4\n 5\n\njulia> stackobs([(1, :a), (2, :b), (3, :c)])\n([1, 2, 3], [:a, :b, :c])\n\njulia> stackobs([rand(256, 256, 3, 1) for _ in 1:10]...) |> size\n(256, 256, 3, 10)\n\njulia> xs = [rand(256, 256, 3, 1) for _ in 1:10];\n\njulia> ys = [rand(256, 256, 1, 1) for _ in 1:10];\n\njulia> data = collect(zip(xs, ys));\n\njulia> stackobs(data) .|> size\n((256, 256, 3, 10), (256, 256, 1, 10))\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.todevice","page":"Home","title":"Apollo.todevice","text":"todevice(x)\n\nCopy x to the GPU.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.unzip","page":"Home","title":"Apollo.unzip","text":"unzip(x::AbstractVector{<:Tuple})\n\nThe reverse of zip.\n\nExample\n\njulia> zip([1, 2, 3], [:a, :b, :c]) |> collect |> unzip\n([1, 2, 3], [:a, :b, :c])\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.vec2array","page":"Home","title":"Apollo.vec2array","text":"vec2array(x::AbstractVector, to::AbstractArray, dim::Int)\n\nReshape the vector x to have the same number of dimensions as to. Missing dimensions  are added as singletons while the dimension corresponding to dim will be filled with the values of x.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.zeros_like","page":"Home","title":"Apollo.zeros_like","text":"zeros_like(x::AbstractArray)\n\nConstruct an array of zeros with the same size and element type as x.\n\n\n\n\n\n","category":"function"},{"location":"#Data-Views","page":"Home","title":"Data Views","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"AbstractView\nJoinedView\nMappedView\nObsView\nTileView\nTransformedView\nZippedView\n\ndropobs\nfilterobs\nmapobs\nrepeatobs\nsampleobs\nshuffleobs\nsplitobs\ntakeobs\nzipobs","category":"page"},{"location":"#Apollo.AbstractView","page":"Home","title":"Apollo.AbstractView","text":"Super type of all iterators.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.JoinedView","page":"Home","title":"Apollo.JoinedView","text":"JoinedView(data...)\n\nAn object that iterates over each element in the iterators given by data as if they were concatenated into a single list.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.MappedView","page":"Home","title":"Apollo.MappedView","text":"MappedView(f, data)\n\nAn iterator which lazily applies f to each element in data when requested.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.ObsView","page":"Home","title":"Apollo.ObsView","text":"ObsView(data, indices)\n\nConstruct an iterator over the elements specified by indices in data.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.TileView","page":"Home","title":"Apollo.TileView","text":"TileView(raster, tilesize::Int; stride=tilesize)\n\nAn object that iterates over tiles cut from a given raster.\n\nParameters\n\nraster: An AbstractRaster or AbstractRasterStack to be cut into tiles.\ntilesize: The size (width and height) of the tiles.\nstride: The horizontal and vertical distance between adjacent tiles.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.TransformedView","page":"Home","title":"Apollo.TransformedView","text":"TransformedView(data, dtype, transform::AbstractTransform).\n\nAn iterator that applied the provided transform to each batch in data. The transform will modify each element according to the specified dtype.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.ZippedView","page":"Home","title":"Apollo.ZippedView","text":"ZippedView(data...)\n\nConstruct an iterator that zips each element of the given subiterators into a Tuple.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.dropobs","page":"Home","title":"Apollo.dropobs","text":"dropobs(data, obs::AbstractVector{Int})\n\nRemove all observations from data whose index corresponds to those in obs.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.filterobs","page":"Home","title":"Apollo.filterobs","text":"filterobs(f, data)\n\nRemove all observations from data for which f returns false.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.mapobs","page":"Home","title":"Apollo.mapobs","text":"mapobs(f, data)\n\nLazily apply the function f to each element in data.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.repeatobs","page":"Home","title":"Apollo.repeatobs","text":"repeatobs(data, n::Int)\n\nCreate a new view which iterates over every element in data n times.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.sampleobs","page":"Home","title":"Apollo.sampleobs","text":"sampleobs([rng=default_rng()], data, n)\n\nRandomly sample n elements from data without replacement. rng may be optionally provided for reproducible results.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.shuffleobs","page":"Home","title":"Apollo.shuffleobs","text":"shuffleobs([rng=default_rng()], data)\n\nRandomly shuffle the elements of data. Provide rng for reproducible results.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.splitobs","page":"Home","title":"Apollo.splitobs","text":"splitobs([rng=default_rng()], data; at=0.8, shuffle=true)\n\nReturn a set of indices that splits the given observations according to the given break points.\n\nArguments\n\ndata: Any type that implements Base.length(). \nat: The fractions at which to split data. \nshuffle: If true, shuffles the indices before splitting. \n\nExample\n\njulia> splitobs(1:100, at=(0.7, 0.2), shuffle=false)\n3-element Vector{Vector{Int64}}:\n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  61, 62, 63, 64, 65, 66, 67, 68, 69, 70]\n [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]\n [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.takeobs","page":"Home","title":"Apollo.takeobs","text":"takeobs(data, obs::AbstractVector{Int})\n\nTake all observations from data whose index corresponds to obs while removing everything else.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.zipobs","page":"Home","title":"Apollo.zipobs","text":"zipobs(data...)\n\nCreate a new iterator where the elements of each iterator in data are returned as a tuple.\n\nExample\n\njulia> zipobs(1:5, 41:45, [:a, :b, :c, :d, :e]) |> collect\n5-element Vector{Any}:\n (1, 41, :a)\n (2, 42, :b)\n (3, 43, :c)\n (4, 44, :d)\n (5, 45, :e)\n\n\n\n\n\n","category":"function"},{"location":"#Transforms","page":"Home","title":"Transforms","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"DType\nImage\nMask\n\nAbstractTransform\nRandomTransform\nComposedTransform\nCrop\nDeNormalize\nFilteredTransform\nNormalize\nRandomCrop\nResample\nTensor\n\napply\ncrop\ndenormalize\nnormalize\nraster\nresample\nresize\ntensor\ntransform\nupsample","category":"page"},{"location":"#Apollo.DType","page":"Home","title":"Apollo.DType","text":"The super-type of all data types.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Image","page":"Home","title":"Apollo.Image","text":"Image([name])\n\nRepresents a type consisting of image data.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Mask","page":"Home","title":"Apollo.Mask","text":"Mask([name])\n\nRepresents an instance of mask data.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.AbstractTransform","page":"Home","title":"Apollo.AbstractTransform","text":"Super type of all transforms.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.RandomTransform","page":"Home","title":"Apollo.RandomTransform","text":"Super type of all random transforms.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.ComposedTransform","page":"Home","title":"Apollo.ComposedTransform","text":"ComposedTransform(transforms...)\n\nApply transforms to the input in the same order as they are given.\n\nExample\n\njulia> r = Raster(rand(256,256, 3), (X,Y,Band));\n\njulia> t = Resample(2.0) |> Tensor();\n\njulia> apply(t, Image(), r, 123) |> size\n(512, 512, 3, 1)\n\njulia> apply(t, Image(), r, 123) |> typeof\nArray{Float32, 4}\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Crop","page":"Home","title":"Apollo.Crop","text":"Crop(size::Int)\nCrop(size::Tuple{Int,Int})\n\nCrop a tile equal to size out of the input array with an upper-left corner at (1,1).\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.DeNormalize","page":"Home","title":"Apollo.DeNormalize","text":"DeNormalize(μ, σ; dim=3)\n\nDenormalize the input array with respect to the specified dimension. Reverses the effect of normalize.\n\nParameters\n\nμ: A Vector of means for each index in dim.\nσ: A Vector of standard deviations for each index in dim.\ndim: The dimension along which to denormalize the input array.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.FilteredTransform","page":"Home","title":"Apollo.FilteredTransform","text":"FilteredTransform(dtype::DType, t::AbstractTransform)\n\nModify the transform t so that it will only be applied to inputs whose type and name matches dtype. The * operator is overloaded for convenience.\n\nExample\n\njulia> r = Raster(rand(256,256, 3), (X,Y,Band));\n\njulia> t = Image(:x2) * Resample(2.0);\n\njulia> apply(t, Image(), r, 123) |> size\n(256, 256, 3)\n\njulia> apply(t, Image(:x2), r, 123) |> size\n(512, 512, 3)\n\njulia> apply(t, Mask(:x2), r, 123) |> size\n(256, 256, 3)\n\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Normalize","page":"Home","title":"Apollo.Normalize","text":"Normalize(μ, σ; dim=3)\n\nNormalize the input array with respect to the specified dimension so that the mean is 0 and the standard deviation is 1.\n\nParameters\n\nμ: A Vector of means for each index in dim.\nσ: A Vector of standard deviations for each index in dim.\ndim: The dimension along which to normalize the input array.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.RandomCrop","page":"Home","title":"Apollo.RandomCrop","text":"RandomCrop(size::Int)\nRandomCrop(size::Tuple{Int,Int})\n\nCrop a randomly placed tile equal to size from the input array.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Resample","page":"Home","title":"Apollo.Resample","text":"Resample(scale)\n\nResample x according to the specified scale. Mask types will always be resampled with :near interpolation, whereas Images will be resampled with  either :bilinear (scale > 1) or :average (scale < 1).\n\nParameters\n\nx: The raster/stack to be resampled.\nscale: The size of the output with respect to the input.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Tensor","page":"Home","title":"Apollo.Tensor","text":"Tensor(;precision=Float32, layerdim=Band)\n\nConvert raster/stack into a tensor with the specified precision. See tensor for more details.\n\nParameters\n\nprecision: Any AbstractFloat to use as the tensor's type (default = Float32).\nlayerdim: RasterStacks will have their layers concatenated along this dimension.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.apply","page":"Home","title":"Apollo.apply","text":"apply(t::AbstractTransform, dtype::DType, data, seed)\n\nApply the transformation t to the data of type dtype with the random seed seed.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.crop","page":"Home","title":"Apollo.crop","text":"crop(x::AbstractArray, size, ul=(1,1))\n\nCrop a tile equal to size out of x with an upper-left corner defined by ul.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.denormalize","page":"Home","title":"Apollo.denormalize","text":"denormalize(x::AbstractArray, μ::AbstractVector, σ::AbstractVector; dim=3)\n\nDenormalize the input array with respect to the specified dimension. Reverses the effect of normalize.\n\nParameters\n\nμ: A Vector of means for each index in dim.\nσ: A Vector of standard deviations for each index in dim.\ndim: The dimension along which to denormalize the input array.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.normalize","page":"Home","title":"Apollo.normalize","text":"normalize(x::AbstractArray, μ::AbstractVector, σ::AbstractVector; dim=3)\n\nNormalize the input array with respect to the specified dimension so that the mean is 0 and the standard deviation is 1.\n\nParameters\n\nμ: A Vector of means for each index in dim.\nσ: A Vector of standard deviations for each index in dim.\ndim: The dimension along which to normalize the input array.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.raster","page":"Home","title":"Apollo.raster","text":"raster(tensor::AbstractArray, dims::Tuple; missingval=0)\n\nRestore the raster dimensions given by dims to the provided tensor. The final dimension of tensor, which is assumed to be the observation dimension, will be dropped.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.resample","page":"Home","title":"Apollo.resample","text":"resample(x::AbstractRaster, scale::AbstractFloat, method=:bilinear)\nresample(x::AbstractRasterStack, scale::AbstractFloat, method=:bilinear)\n\nResample x according to the given scale and method.\n\nParameters\n\nx: The raster/stack to be resampled.\nscale: The size of the output with respect to the input.\nmethod: One of :near, :bilinear, :cubic, :cubicspline, :lanczos, or :average.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.resize","page":"Home","title":"Apollo.resize","text":"resize(x::AbstractRaster, newsize, method=:bilinear)\nresize(x::AbstractRasterStack, newsize, method=:bilinear)\n\nResize the raster/stack x to newsize under the specified method.\n\nParameters\n\nx: The array to be resized.\nnewsize: The width and height of the output as a tuple.\nmethod: One of :near, :bilinear, :cubic, :cubicspline, :lanczos, or :average.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.tensor","page":"Home","title":"Apollo.tensor","text":"tensor([precision], xs...; kwargs...)\ntensor([precision], x::AbstractArray; kwargs...)\ntensor([precision], x::AbstractRasterStack; layerdim=Band)\n\nConvert one or more arrays to a tensor with an element type of precision. AbstractRasters will be reshaped as necessary to enforce a dimension order of (X,Y,Z,Band,Ti) before adding an observation dimension. Multiple arrays will be  concatenated along the observation dimension after being converted to tensors.\n\nParameters\n\nprecision: Any AbstractFloat to use as the tensor's type (default = Float32).\nx: One or more AbstractArrays to be turned into tensors.\nlayerdim: AbstractRasterStacks will have their layers concatenated along this dimension\n\nbefore being turned into tensors.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.transform","page":"Home","title":"Apollo.transform","text":"transform(t::AbstractTransform, dtype::DType, x)\ntransform(t::AbstractTransform, dtypes::Tuple, x::Tuple)\n\nApply the transformation t to the input x with data type dtype.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.upsample","page":"Home","title":"Apollo.upsample","text":"upsample(x::AbstractArray, scale, method=:bilinear)\n\nUpsample the array x according to the given scale and method.\n\nParameters\n\nx: The array to be upsampled.\nscale: The size of the output with respect to the input.\nmethod: One of :linear, :bilinear, :trilinear, or :nearest.\n\n\n\n\n\n","category":"function"},{"location":"#Metrics","page":"Home","title":"Metrics","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"AbstractMetric\nClassificationMetric\nRegressionMetric\nMetric\nAccuracy\nMIoU\nMSE\nLoss\ncompute\nevaluate\ninit\nname\nreset!\nupdate\nupdate!","category":"page"},{"location":"#Apollo.AbstractMetric","page":"Home","title":"Apollo.AbstractMetric","text":"Abstract supertype of all metrics.\n\nMetrics are measures of a model's performance, such as loss, accuracy, or squared error.\n\nEach metric must implement the following interface:\n\nname(::Type{metric}): Returns the human readable name of the metric.\ninit(metric): Returns the initial state of the metric as a NamedTuple.\nupdate(metric, state, ŷ, y): Returns the new state given the previous state and a batch.\ncompute(metric, state): Computes the metric's value for the current state.\n\nExample Implementation\n\nstruct Accuracy <: ClassificationMetric end\n\nname(::Type{Accuracy}) = \"accuracy\"\n\ninit(::Accuracy) = (correct=0, total=0)\n\nfunction update(::Accuracy, state, ŷ, y)\n    return (correct = state.correct + sum(ŷ .== y), total = state.total + length(ŷ))\nend\n\ncompute(::Accuracy, state) = state.correct / max(state.total, 1)\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.ClassificationMetric","page":"Home","title":"Apollo.ClassificationMetric","text":"Super type of all classification metrics.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.RegressionMetric","page":"Home","title":"Apollo.RegressionMetric","text":"Super type of all regression metrics.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Metric","page":"Home","title":"Apollo.Metric","text":"Metric(measure::AbstractMetric)\n\nMetric objects are used to store the state for a given AbstractMetric.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Accuracy","page":"Home","title":"Apollo.Accuracy","text":"Accuracy()\n\nMeasures the model's overall accuracy as correct / total.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.MIoU","page":"Home","title":"Apollo.MIoU","text":"MIoU(classes::Vector{Int})\n\nMean Intersection over Union (MIoU) is a measure of the overlap between a prediction and a label. This measure is frequently used for segmentation models.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.MSE","page":"Home","title":"Apollo.MSE","text":"MSE()\n\nTracks Mean Squared Error (MSE) as ((ŷ - y) .^ 2) / length(y).\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Loss","page":"Home","title":"Apollo.Loss","text":"Loss(loss::Function)\n\nTracks the average model loss as total_loss / steps\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.compute","page":"Home","title":"Apollo.compute","text":"compute(m::AbstractMetric, state)\n\nCompute the performance measure from the current state.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.evaluate","page":"Home","title":"Apollo.evaluate","text":"evaluate(model, data, measures...)\n\nEvaluate the model's performance on the provided data.\n\nParameters\n\nmodel: A callable that takes a single batch from data and returns a tuple of the form (ŷ, y).\ndata: An iterable of (x, y) values.\nmeasures: A set of AbstractMetrics to use for evaluating model.\n\nReturns\n\nA NamedTuple containing the performance metrics for the given model.\n\nExample\n\nevaluate(DataLoader((xsampler, ysampler)), Accuracy(), MIoU(2)) do (x, y)\n    ŷ = model(x) |> Flux.sigmoid\n    return (ŷ, y)\nend\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.init","page":"Home","title":"Apollo.init","text":"init(m::AbstractMetric)\n\nReturns the initial state of the performance measure, which will be subsequently updated for each mini-batch of labels and predictions.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.name","page":"Home","title":"Apollo.name","text":"name(m::AbstractMetric)\nname(m::Metric)\n\nHuman readable name of the given performance measure.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.reset!","page":"Home","title":"Apollo.reset!","text":"reset!(metric::Metric)\n\nReset the metric's state.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.update","page":"Home","title":"Apollo.update","text":"update(m::AbstractMetric, state, ŷ, y)\n\nReturn the new state for the given batch of labels and predictions.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.update!","page":"Home","title":"Apollo.update!","text":"update!(metric::Metric, ŷ, y)\n\nUpdate the metric state for the next batch of labels and predictions.\n\n\n\n\n\n","category":"function"},{"location":"#Performance-Tracking","page":"Home","title":"Performance Tracking","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Tracker\nMetricLogger\nMax\nMin\n\nbest_epoch\ncurrent_epoch\nepoch!\nprintscores\nscores\nstep!","category":"page"},{"location":"#Apollo.Tracker","page":"Home","title":"Apollo.Tracker","text":"Tracker(metrics...)\n\nAn object to track one or more metric values over the course of a training run. Each metric can be either an AbstractMetrics or a name => metric pair. In the first case, the default name of the provided metric will be used, while the second allows us to choose an arbitrary name.\n\nTracker implements the Tables.jl interface, allowing it to be used as a table source.\n\nExample 1\n\njulia> tracker = Tracker(Accuracy(), MIoU([0,1]));\n\njulia> step!(tracker, [0, 0, 1, 0], [0, 1, 1, 0]);  # update all metrics\n\njulia> scores(tracker)\n(epoch = 1, accuracy = 0.75, MIoU = 0.5833333730697632)\n\njulia> epoch!(tracker);\n\njulia> tracker\nTracker(current_epoch=2)\n┌───────┬──────────┬──────────┐\n│ epoch │ accuracy │     MIoU │\n│ Int64 │  Float64 │  Float64 │\n├───────┼──────────┼──────────┤\n│     1 │     0.75 │ 0.583333 │\n│     2 │      0.0 │      0.0 │\n└───────┴──────────┴──────────┘\n\nExample 2\n\njulia> tracker = Tracker(\"train_acc\" => Accuracy(), \"val_acc\" => Accuracy());\n\njulia> step!(tracker, \"train_acc\", [0, 0, 1, 0], [0, 1, 1, 0]);  # specify metric to update\n\njulia> step!(tracker, r\"val_\", [0, 0, 0, 0], [0, 1, 1, 0]);  # update metrics matching regex\n\njulia> tracker\nTracker(current_epoch=1)\n┌───────┬───────────┬─────────┐\n│ epoch │ train_acc │ val_acc │\n│ Int64 │   Float64 │ Float64 │\n├───────┼───────────┼─────────┤\n│     1 │      0.75 │     0.5 │\n└───────┴───────────┴─────────┘\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.MetricLogger","page":"Home","title":"Apollo.MetricLogger","text":"MetricLogger(metrics...; prefix=\"\")\n\nAn object to track one or more metrics. Each metric is associated with a unique name,  which defaults to name(metric). This can be overriden by providing a name => metric pair. The prefix keyword adds a constant prefix string to every name.\n\nExample 1\n\njulia> md = MetricLogger(Accuracy(), MIoU([0,1]); prefix=\"train_\");\n\njulia> step!(md, [0, 0, 1, 0], [0, 0, 1, 1]);\n\njulia> md\nMetricLogger(train_accuracy=0.75, train_MIoU=0.5833334)\n\njulia> step!(md, [0, 0, 1, 1], [0, 0, 1, 1]);\n\njulia> md\nMetricLogger(train_accuracy=0.875, train_MIoU=0.775)\n\njulia> reset!(md)\n\njulia> md\nMetricLogger(train_accuracy=0.0, train_MIoU=0.0)\n\nExample 2\n\njulia> md = MetricLogger(\"train_acc\" => Accuracy(), \"val_acc\" => Accuracy());\n\njulia> step!(md, \"train_acc\", [0, 1, 1, 0], [1, 1, 1, 0]);  # update train_acc\n\njulia> step!(md, r\"val_\", [1, 1, 1, 0], [1, 1, 1, 0]);  # update metrics matching regex\n\njulia> md\nMetricLogger(train_acc=0.75, val_acc=1.0)\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Max","page":"Home","title":"Apollo.Max","text":"Max(metric::String)\n\nRepresents an ordering of largest to smallest for the given metric.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Min","page":"Home","title":"Apollo.Min","text":"Min(metric::String)\n\nRepresents an ordering of smallest to largest for the given metric.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.best_epoch","page":"Home","title":"Apollo.best_epoch","text":"best_epoch(tracker::Tracker, order::Order)\n\nReturns the best epoch in tracker according to the given order.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.current_epoch","page":"Home","title":"Apollo.current_epoch","text":"current_epoch(tracker::Tracker)\n\nThe current epoch in tracker.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.epoch!","page":"Home","title":"Apollo.epoch!","text":"epoch!(tracker::Tracker)\n\nStore the metric value for the current epoch and reset the state.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.printscores","page":"Home","title":"Apollo.printscores","text":"printscores(t::Tracker; epoch=current_epoch(t), metrics=r\".*\")\n\nReturn the metric scores for the provided epoch as a pretty-printed string.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.scores","page":"Home","title":"Apollo.scores","text":"scores(t::Tracker; epoch=current_epoch(t), metrics=keys(t.metrics))\n\nReturn the metric scores for the provided epoch.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.step!","page":"Home","title":"Apollo.step!","text":"step!(x::MetricLogger, ŷ, y)\nstep!(x::MetricLogger, metric::String, ŷ, y)\nstep!(x::MetricLogger, metric::Regex, ŷ, y)\n\nUpdate the metric for the current epoch using the provided prediction/label pair.\n\n\n\n\n\nstep!(tracker::Tracker, ŷ, y)\nstep!(tracker::Tracker, metric::String, ŷ, y)\nstep!(tracker::Tracker, metric::Regex, ŷ, y)\n\nUpdate the metric for the current epoch using the provided prediction/label pair.\n\n\n\n\n\n","category":"function"},{"location":"#Layers","page":"Home","title":"Layers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ConvBlock\nLSTM\nSeparableConv","category":"page"},{"location":"#Apollo.ConvBlock","page":"Home","title":"Apollo.ConvBlock","text":"ConvBlock(filter, in, out, σ; depth=2, batch_norm=true)\n\nA block of convolutional layers with optional batch normalization.\n\nParameters\n\nfilter: The size of the filter to use for each layer.\nin: The number of input features.\nout: The number of output features.\nσ: The activation function to apply after each convolution.\ndepth: The number of successive convolutions in the block.\nbatch_norm: Applies batch normalization after each convolution if true.\n\n\n\n\n\n","category":"function"},{"location":"#Apollo.LSTM","page":"Home","title":"Apollo.LSTM","text":"LSTM(in => out)\n\nA stateless pixel-wise LSTM layer that expects an input tensor with shape WHCLN, where W is image width, H is image height, C is image channels, L is sequence length, and N is  batch size. The LSTM module will be applied to each pixel in the sequence from first to last  with all but the final output discarded.\n\nParameters\n\nin: The number of input features.\nout: The number of output features.\n\nExample\n\njulia> m = Apollo.LSTM(2=>32);\n\njulia> x = rand(Float32, 64, 64, 2, 9, 4);\n\njulia> m(x) |> size\n(64, 64, 32, 4)\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.SeparableConv","page":"Home","title":"Apollo.SeparableConv","text":"SeparableConv(filter, in, out, σ=Flux.relu)\n\nA separable convolutional layer consists of a depthwise convolution followed by a 1x1 convolution.\n\nParameters\n\nfilter: The size of the filter as a tuple.\nin: The number of input features.\nout: The number of output features.\nσ: The activation function to apply following the 1x1 convolution.\n\n\n\n\n\n","category":"function"},{"location":"#Inputs","page":"Home","title":"Inputs","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"AbstractInput\nSingle\nSeries\nbuild_input","category":"page"},{"location":"#Apollo.AbstractInput","page":"Home","title":"Apollo.AbstractInput","text":"Super type of all input layers. Objects implementing this interface need to provide an instance of the build_input method.\n\nExample Implementation\n\nstruct Single <: AbstractInput\n    channels::Int\n    batch_norm::Bool\nend\n\nSingle(;channels=3, batch_norm=true) = Single(channels, batch_norm)\n\nfunction build_input(x::Single, out_features::Int)\n    ConvBlock((3,3), x.channels, out_features, Flux.relu, batch_norm=x.batch_norm)\nend\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Single","page":"Home","title":"Apollo.Single","text":"Single(;channels=3, batch_norm=true)\n\nDefines an input as a singular image. Low-level features are extracted by two consecutive 3x3 convolutions.\n\nParameters\n\nchannels: The number of bands in the image.\nbatch_norm: If true, batch normalization will be applied after the convolutional layers.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.Series","page":"Home","title":"Apollo.Series","text":"Series(;channels=3, batch_norm=true)\n\nDefines an input as a series of images, where each image contains the same number of channels. The temporal features will first be extracted by a pixel-wise LSTM module, which will then be subjected to two consecutive 3x3 convolutions.\n\nParameters\n\nchannels: The number of bands in each image.\nbatch_norm: If true, batch normalization will be applied after the convolutional layers.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.build_input","page":"Home","title":"Apollo.build_input","text":"build_input(x::AbstractInput, out_features::Int)\n\nConstructs an input layer based on the provided AbstractInput configuration and the specified number of output features.\n\nArguments\n\nx: An AbstractInput object specifying the configuration and properties of the input layer to be built, such as input type (e.g., single-channel, multi-channel).\nout_features: The number of output features (or channels) that the input layer should produce. This determines the dimensionality of the output from the input layer.\n\nReturns\n\nA Flux layer representing the constructed input module, which can be used as the starting point of a neural network architecture.\n\nExample\n\ninput_layer = build_input(Single(channels=4), 64)\n\n\n\n\n\n","category":"function"},{"location":"#Encoders","page":"Home","title":"Encoders","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"AbstractEncoder\nResNet18\nResNet34\nResNet50\nResNet101\nResNet152\nStandardEncoder\nbuild_encoder","category":"page"},{"location":"#Apollo.AbstractEncoder","page":"Home","title":"Apollo.AbstractEncoder","text":"AbstractEncoder{F}\n\nSuper type of all encoder models. The type parameter F denotes the number of features produced by each block of the encoder.\n\nExample Implementation\n\nstruct ResNet18{W} <: AbstractEncoder{(64,64,128,256,512)}\n    weights::W\n    ResNet18(;weights=nothing) = new{typeof(weights)}(weights)\nend\n\nbuild_encoder(x::ResNet18) = resnet(18, x.weights)\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.ResNet18","page":"Home","title":"Apollo.ResNet18","text":"ResNet18(;weights=nothing)\n\nConstruct a ResNet18 encoder with the specified initial weights.\n\nParameters\n\nweights: Either :ImageNet or nothing.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.ResNet34","page":"Home","title":"Apollo.ResNet34","text":"ResNet34(;weights=nothing)\n\nConstruct a ResNet34 encoder with the specified initial weights.\n\nParameters\n\nweights: Either :ImageNet or nothing.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.ResNet50","page":"Home","title":"Apollo.ResNet50","text":"ResNet50(;weights=nothing)\n\nConstruct a ResNet50 encoder with the specified initial weights.\n\nParameters\n\nweights: Either :ImageNet or nothing.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.ResNet101","page":"Home","title":"Apollo.ResNet101","text":"ResNet101(;weights=nothing)\n\nConstruct a ResNet101 encoder with the specified initial weights.\n\nParameters\n\nweights: Either :ImageNet or nothing.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.ResNet152","page":"Home","title":"Apollo.ResNet152","text":"ResNet152(;weights=nothing)\n\nConstruct a ResNet152 encoder with the specified initial weights.\n\nParameters\n\nweights: Either :ImageNet or nothing.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.StandardEncoder","page":"Home","title":"Apollo.StandardEncoder","text":"StandardEncoder(;batch_norm=true)\n\nConstruct a standard UNet encoder with no pre-trained weights.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.build_encoder","page":"Home","title":"Apollo.build_encoder","text":"build_encoder(encoder::AbstractEncoder)\n\nConstructs an encoder model based on the provided AbstractEncoder configuration.\n\nParameters\n\nencoder: An AbstractEncoder object specifying the architecture and configuration of the encoder to be built.\n\nReturns\n\nA standard Flux.Chain layer containing each block of the encoder. The returned encoder is ready to be integrated into more complex architectures like U-Net or used as a standalone feature extractor.\n\nExample\n\nencoder = build_encoder(ResNet50(weights=:ImageNet))\n\n\n\n\n\n","category":"function"},{"location":"#Models","page":"Home","title":"Models","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Classifier\nDeeplabV3\nR2UNet\nSSC_CNN\nUNet","category":"page"},{"location":"#Apollo.Classifier","page":"Home","title":"Apollo.Classifier","text":"Classifier(;input=Single(), encoder=ResNet50(), nclasses=2)\n\nConstruct an image classifier from the provided encoder.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.DeeplabV3","page":"Home","title":"Apollo.DeeplabV3","text":"DeeplabV3(encoder::AbstractEncoder; channels=3, nclasses=1)\n\nConstruct a DeeplabV3+ model.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.R2UNet","page":"Home","title":"Apollo.R2UNet","text":"R2UNet(in_features, n_classes; t=2, batch_norm=false)\n\nConstruct an R2-UNet model.\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.SSC_CNN","page":"Home","title":"Apollo.SSC_CNN","text":"SSC_CNN(;width=128)\n\nConstruct an SSC_CNN model (Nguyen et al.)  to sharpen the 20m Sentinel-2 bands to a resolution of 10m.\n\nParameters\n\nwidth: The number of features to use in each block (default = 128).\n\n\n\n\n\n","category":"type"},{"location":"#Apollo.UNet","page":"Home","title":"Apollo.UNet","text":"UNet(;input=Single(), encoder=StandardEncoder(), nclasses=1, activation=identity, batch_norm=true)\n\nConstruct a UNet model.\n\nKeywords\n\ninput: The input block, which defaults to two convolutional layers as with standard UNet.\nencoder: The encoder to use for the UNet model. Defaults to the standard encoder.\nnclasses: The number of output channels produced by the head.\nactivation: The activation to apply after the final convolutional layer.\nbatch_norm: Use batch normalization after each convolutional layer (default=true).\n\n\n\n\n\n","category":"type"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
