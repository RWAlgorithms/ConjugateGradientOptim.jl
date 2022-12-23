

### linesearch configs.

# parameters in eqn 8 and 9 of (Yuan 2019).
# struct LinesearchWolfe{T}

# end


### generic linesearch intermediate quantities.

struct LineSearchContainer{T} # all arrays have same size.
    xp::Vector{T}  # proposed iterate.
    df_xp::Vector{T}
    
    x::Vector{T} # current iterate.
    u::Vector{T} # search direction.
end

function LineSearchContainer(::Type{T}, N::Int) where T <: AbstractFloat

    return LineSearchContainer(
        Vector{T}(undef, N),
        Vector{T}(undef, N),
        Vector{T}(undef, N),
        Vector{T}(undef, N),
    )
end


#### generic linesearch routines.

function evalϕdϕ!(
    xp::Vector{T},
    df_xp::Vector{T},
    fdf!,
    a::T, # step size.
    x::Vector{T}, # current iterate.
    u::Vector{T}, # search direction.
    ) where T

    # update iterate on a line.
    for i in eachindex(x)
        xp[i] = x[i] + a * u[i]
    end

    # evaluate the objective and its gradient restricted to the line.
    ϕ_xp = fdf!(df_xp, xp)
    dϕ_xp = dot(df_xp, u)

    return ϕ_xp, dϕ_xp
end
