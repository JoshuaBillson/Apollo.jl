struct Pipeline{F,D}
    f::F
    data::D
end

Base.length(x::Pipeline{F,<:Tuple}) where F = map(length, x.data) |> minimum

Base.getindex(x::Pipeline, i::Int) = x.f(x.data[i])
Base.getindex(x::Pipeline{<:Function,<:Tuple}, i::Int) = x.f(map(x -> getindex(x, i), x.data))
Base.getindex(x::Pipeline, i::AbstractVector) = _stack(map(j -> getindex(x, j), i)...)
function Base.getindex(x::Pipeline, i::AbstractVector)
    return @pipe map(j -> getindex(x, j), i) |> _unzip(_) |> map(x -> _stack(x...), _)
end