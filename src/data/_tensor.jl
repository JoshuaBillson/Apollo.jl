"""
abstract type TDim end
struct W <: TDim end
struct H <: TDim end
struct C <: TDim end
struct L <: TDim end
struct N <: TDim end
"""
@enum TDim W H C L N

struct DimIndex{D,I}
    dim::D
    index::I
end

Width(i) = DimIndex(W, i)
Height(i) = DimIndex(H, i)
Channel(i) = DimIndex(C, i)
Length(i) = DimIndex(L, i)
Obs(i) = DimIndex(N, i)

abstract type TShape end
struct WHCN <: TShape end
struct WHCLN <: TShape end

struct Tensor{T<:Real,N,D,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A

    function Tensor(dims::Tuple, array::AbstractArray{T,N}) where {T, N}
        return new{T,N,dims,typeof(array)}(array)
    end
end

Tensor(::Type{WHCN}, x::AbstractArray{<:Any,4}) = Tensor((W, H, C, N), x)
Tensor(::Type{WHCN}, x::AbstractArray{<:Any,3}) = Tensor((W, H, C, N), MLUtils.unsqueeze(x, dims=4))
Tensor(::Type{WHCLN}, x::AbstractArray{<:Any,5}) = Tensor((W, H, C, L, N), x)
Tensor(::Type{WHCLN}, x::AbstractArray{<:Any,4}) = Tensor((W, H, C, L, N), MLUtils.unsqueeze(x, dims=5))
Tensor(::Type{T}, x) where {T <: TShape} = Tensor(T, tensor(T, x))
Tensor(::Type{T}, xs...) where {T <: TShape} = Tensor(T, tensor(T, xs...))

tensor(::Type{T}, xs::AbstractVector; kwargs...) where {T <: TShape} = tensor(T, xs...; kwargs...)
function tensor(::Type{T}, xs...; precision=:f32) where {T <: TShape}
    return @pipe map(x -> tensor(T, x; precision=precision), xs) |> _stack(_...)
end
function tensor(::Type{T}, x::AbstractDimStack; precision=:f32) where {T <: TShape}
    return tensor(T, _catlayers(T, x); precision=precision)
end
function tensor(::Type{T}, x::AbstractDimArray; precision=:f32) where {T <: TShape}
    if !hasdim(x, Band)
        return tensor(T, putdim(x, Band); precision=precision)
    end
    rdims = _rdims(T)
    _dims_match(x, rdims)
    return _tensor(x, rdims, precision)
end

raster(tensor::Tensor{T,TN,D}, rdims::Tuple) where {T,TN,D} = raster(tensor, D, rdims)
raster(tensor::AbstractArray, ::Type{WHCN}, rdims::Tuple) = raster(tensor, (W,H,C,N), rdims)
raster(tensor::AbstractArray, ::Type{WHCLN}, rdims::Tuple) = raster(tensor, (W,H,C,L,N), rdims)
function raster(tensor::AbstractArray{T,TN}, tdims::Tuple, rdims::Tuple) where {T,TN}
    (TN != length(tdims)) && throw(ArgumentError("Provided $(length(tdims)) dimensions for $TN dimensional tensor!"))
    if N in tdims
        obs_index = findfirst(==(N), tdims)
        if size(tensor, obs_index) == 1 
            return raster(selectdim(tensor, obs_index, 1), filter(!=(N), tdims), rdims)
        end
        throw(ArgumentError("Tensor contains multiple observations!"))
    end
    (TN != length(rdims)) && throw(ArgumentError("Provided $(length(rdims)) raster dimensions for $TN dimensional tensor!"))
    newdims = dims(rdims, map(_tdim_to_rdim, tdims))
    return Raster(tensor, newdims)
end

Base.size(x::Tensor, dim::Integer) = Base.size(x.data, dim)
Base.size(x::Tensor) = Base.size(x.data)
function Base.size(x::Tensor{T,TN,D}, dim::TDim) where {T,TN,D}
    size(x.data, _dim_index(x, dim))
end

Base.getindex(x::Tensor, args...) = Base.getindex(x.data, args...)
Base.getindex(x::Tensor, dim::DimIndex) = selectdim(x, dim.dim, dim.index)
Base.getindex(x::Tensor, dims::Vararg{DimIndex}) = reduce((acc, i) -> selectdim(acc, i.dim, i.index), dims, init=x)

Base.selectdim(t::Tensor, dim::Integer, i) = selectdim(t.data, dim, i)
function Base.selectdim(t::Tensor{T,N,D,A}, dim::TDim, i::Int) where {T,N,D,A}
    return Tensor(filter(!=(dim), D), selectdim(t.data, _dim_index(t, dim), i))
end
function Base.selectdim(t::Tensor{T,N,D}, dim::TDim, i::AbstractVector) where {T, N, D}
    return Tensor(D, selectdim(t.data, _dim_index(t, dim), i))
end

function Base.permutedims(t::Tensor{T,N,D}, dims) where {T,N,D}
    perm = [_dim_index(t, dim) for dim in dims]
    return Tensor(dims, permutedims(t.data, perm))
end

function Base.show(io::IO, ::MIME"text/plain", t::Tensor{T,N,D,A}) where {T,N,D,A}
    _size = join(size(t), "×")
    #_dims = map(typeof, t.dims)
    printstyled(io, "$_size Tensor{$T,$N}(dims=$D)")
#    Base.(io, "text/plain", t.data)
end

function _dim_index(::Tensor{T,N,D}, dim::TDim) where {T,N,D}
    d = findfirst(==(dim), D)
    if isnothing(d)
        throw(ArgumentError("Tensor does not contain dimension `$dim`!"))
    end
    return d
end

_rdims(::Type{WHCN}) = (X,Y,Band)
_rdims(::Type{WHCLN}) = (X,Y,Band,Ti)

_catlayers(::Type{WHCN}, x::AbstractDimStack) = catlayers(x, Band)
_catlayers(::Type{WHCLN}, x::AbstractDimStack) = catlayers(x, Ti)

function _tdim_to_rdim(dim::TDim)
    @match dim begin
        $W => X
        $H => Y
        $C => Band
        $L => Ti
    end
end

function _tensor(x::AbstractDimArray{T,N}, dims::Tuple, precision) where {T,N}
    @pipe _permute(x, dims).data |> 
    MLUtils.unsqueeze(_, dims=N+1) |> 
    _precision(_, precision)
end