
# quantities that don't change throughout the entire CG algorithm.
struct LineSearchConfig{FT,T}
    fdf!::FT
    θ::T
    max_iters::Int
end

# mutable struct LineSearchIntermediates{T}

#     ϕ_0::T
#     dϕ_0::T

#     a::T
#     b::T

#     ϕ_a::T
#     dϕ_a::T

#     ϕ_b::T
#     dϕ_b::T

#     status::Bool
# end

# quantities that don't change throughout a line search.
struct LineSearchContainer{T} # all arrays have same size.
    xp::Vector{T}  # proposed iterate.
    df_xp::Vector{T}
    
    x::Vector{T} # current iterate.
    u::Vector{T} # search direction.

    #ϕ_xp::T
    #dϕ_xp::T
    
    #ϕ_0::T
    #dϕ_0::T
end

mutable struct LineSearchEval{T}
    ϕ_a::T
    dϕ_a::T

    ϕ_b::T
    dϕ_b::T
end

function LineSearchContainer(::Type{T}, N::Int) where T <: AbstractFloat

    return LineSearchContainer(
        Vector{T}(undef, N),
        Vector{T}(undef, N),
        Vector{T}(undef, N),
        Vector{T}(undef, N),
        #convert(T, NaN),
        #convert(T, NaN),
    )
end

function updatecontainer!(
    info::LineSearchContainer{T},
    x::Vector{T},
    u::Vector{T},
    #f_x::T,
    #df_x::Vector{T},
    ) where T

    #
    info.x[:] = x
    info.u[:] = u
    #info.ϕ_0 = f_x
    #info.dϕ_0 = dot(df_x, u)

    return nothing
end

function evalϕdϕ!(
    info::LineSearchContainer{T},
    fdf!,
    a::T, # step size.
    #x::Vector{T}, 
    #u::Vector{T}, 
    ) where T

    return evalϕdϕ!(info.xp, info.df_xp, fdf!, a, info.x, info.u)
end

function evalϕdϕ!(
    #info::LineSearchContainer{T},
    xp::Vector{T},
    df_xp::Vector{T},
    fdf!,
    a::T, # step size.
    x::Vector{T}, # current iterate.
    u::Vector{T}, # search direction.
    ) where T

    # parse.
    #xp, df_xp, ϕ_xp, dϕ_xp, ϕ_0, dϕ_0 = info.xp, info.df_xp, info.ϕ_xp, info.dϕ_xp, info.ϕ_0, info.dϕ_0

    # update iterate on a line.
    for i in eachindex(x)
        xp[i] = x[i] + a * u[i]
    end

    # evaluate the cost and its gradient restricted to the line.
    ϕ_xp = fdf!(df_xp, xp)
    dϕ_xp = dot(df_xp, u)

    return ϕ_xp, dϕ_xp
end