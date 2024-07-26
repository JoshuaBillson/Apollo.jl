struct WaldProtocol{D1,D2}
    traindata::D1
    testdata::D2
end

function WaldProtocol(lr::HasDims, hr::HasDims)
    scale = _tilesize(hr) ./ _tilesize(lr)
    @assert all(isinteger, scale)
    @assert scale[1] == scale[2]
    lr_down = resample(lr, 1 / scale[1], :average)
    hr_down = resample(hr, 1 / scale[1], :average)
    return WaldProtocol(
        _wald_pipe(lr_down, hr_down, scale[1], 32),
        _wald_pipe(lr, hr, scale[1], 64)
    )
end

_wald_pipe(lr, hr, scale, tilesize) = _wald_pipe(lr, hr, Int(scale), Int(tilesize))
function _wald_pipe(lr, hr, scale::Int, tilesize::Int)
    println(scale, " ", tilesize)
    lr_pipe = mapobs(x -> resample(x, scale, :bilinear), TileSampler(lr, tilesize÷scale, stride=tilesize÷scale÷2))
    hr_pipe = TileSampler(hr, tilesize, stride=tilesize÷2)
    return zipobs(lr_pipe, hr_pipe)
end

function getobs(x::WaldProtocol, i::Int, partition::Symbol)
    @match partition begin
        :train => x.traindata[i]
        :test => x.testdata[i]
        _ => throw(ArgumentError("WaldProtocol only supports the :train and :test partitions!"))
    end
end

function nobs(x::WaldProtocol, partition::Symbol)
    @match partition begin
        :train => length(x.traindata)
        :test => length(x.testdata)
        _ => throw(ArgumentError("WaldProtocol only supports the :train and :test partitions!"))
    end
end

function getdata(x::WaldProtocol, partition::Symbol)
    @match partition begin
        :train => mapobs(i -> getobs(x, i, :train), 1:nobs(x, :train))
        :test => mapobs(i -> getobs(x, i, :test), 1:nobs(x, :test))
        _ => throw(ArgumentError("WaldProtocol only supports the :train and :test partitions!"))
    end

end
