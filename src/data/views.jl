struct MappedView{F,D}
    f::F
    data::D
end

Base.length(x::MappedView) = length(x.data)
Base.length(x::MappedView{<:Function,<:Tuple}) = map(length, x.data) |> minimum

Base.getindex(x::MappedView, i::Int) = x.f(x.data[i])
Base.getindex(x::MappedView{<:Function,<:Tuple}, i::Int) = x.f(map(data -> data[i], x.data))
Base.getindex(x::MappedView, i::AbstractVector) = map(j -> getindex(x, j), i) |> _stack

Base.iterate(x::MappedView, state=1) = state > length(x) ? nothing : (x[state], state+1)

struct JoinedView{D}
    data::D
    JoinedView(data) = JoinedView((data,))
    JoinedView(data...) = JoinedView(data)
    JoinedView(data::D) where {D <: Tuple} = new{D}(data)
end

Base.length(x::JoinedView) = map(length, x.data) |> sum

Base.getindex(x::JoinedView, i::AbstractVector) = _stack([getindex(x, j) for j in i])
function Base.getindex(x::JoinedView, i::Int)
    (i > length(x) || i < 1) &&  throw(BoundsError(x, i)) 
    lengths = map(length, x.data)
    for (j, len) in enumerate(cumsum(lengths))
        if i <= len
            offset = len - first(lengths)
            return  x.data[j][i-offset]
        end
    end
end

Base.iterate(x::JoinedView, state=1) = state > length(x) ? nothing : (x[state], state+1)