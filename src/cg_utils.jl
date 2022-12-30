
function updatedir!(
    β_config::CGβConfig,
    u::Vector{T},
    df_x::Vector{T},
    a::T,
    ) where T

    @assert length(u) == length(df_x)

    for i in eachindex(u)
        u[i] = -df_x[i] + a*u[i]
    end

    return nothing
end

function updatedir!(
    β_config::QNβConfig,
    u::Vector{T},
    df_x::Vector{T},
    a::T,
    ) where T

    @assert length(u) == length(df_x)

    # here.
    # for i in eachindex(u)
    #     u[i] = -df_x[i] + a*u[i]
    # end

    return nothing
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


