### generic core routines.
function updatedir!(
    u::Vector{T},
    df_x::Vector{T},
    β::T,
    ) where T <: AbstractFloat

    @assert length(u) == length(df_x)

    for i in eachindex(u)
        u[i] = -df_x[i] + β*u[i]
    end

    return nothing
end

function initializeβ(::Type{T}, β_config::CGβConfig) where T
    return zeros(T, 1)
end

# conjugate gradient.
function initializeLineSearchContainer!(
    info::LineSearchContainer{T},
    β_config::CGβConfig,
    df_x::Vector{T},
    x::Vector{T},
    ) where T

    info.u[:] = -df_x # search direction.
    info.x[:] = x
    info.xp[:] = x
    info.df_xp[:] = df_x
    
    return nothing
end



#### various types of conjugate gradient β values.
# See `ConjugateGradientOptim.jl`  for full reference information.

# TODO: make these updates non-allocating.


# (Yuang 2019): modified Hager-Zhang, to have trust region behavior.
struct YuanWangSheng{T} <: CGβConfig
    μ::T
end

function setupYuanWangSheng(μ::T)::YuanWangSheng{T} where T

    @assert zero(T) < μ < one(T)

    return YuanWangSheng(μ)
end

# sec 2.2 of (Yuan 2019).
function getβ(
    β_config::YuanWangSheng{T},
    g_next::Vector{T},
    g::Vector{T},
    u::Vector{T},
    #μ::T, # 0 < μ < 1
    )::T where T

    # parse.
    μ = β_config.μ

    # calculation.
    y = g_next - g
    
    R1 = μ*norm(u)*norm(y)
    R2 = dot(u,y)
    R3 = 2*dot(y,y)*dot(u,g_next)/dot(y,g_next)
    R = max(R1, R2, R3)
    #R = R2 # force the Hager–Zhang case; see sec 2.2 of (Yuan 2019).

    tmp2 = g_next ./ R

    m = 2*dot(y,y)/R
    tmp1 = y - m .* u

    β_MN = dot(tmp1, tmp2)

    return β_MN
end


# The HagerZhang linesearch from Algorithm 851 is not implemented as of now.
struct HagerZhang <: CGβConfig end

# sec 2.2 of (Yuan 2019).
# This is the YuanWangSheng version, but set R = R2.
function getβ(
    β_config::HagerZhang,
    g_next::Vector{T},
    g::Vector{T},
    u::Vector{T},
    #μ::T, # 0 < μ < 1
    )::T where T

    # calculation.
    y = g_next - g
    
    R = dot(u,y)
    
    tmp2 = g_next ./ R

    m = 2*dot(y,y)/R
    tmp1 = y - m .* u

    β_MN = dot(tmp1, tmp2)

    return β_MN
end

# Don't use Hestenses-Stiefel because it can yield ascent search directions.
# struct HestensesStiefel <: CGβConfig end

# # sec 1 from (Yuan 2021).
# function getβ(
#     β_config::HestensesStiefel,
#     g_next::Vector{T},
#     g::Vector{T},
#     u::Vector{T},
#     )::T where T

#     y = g_next - g
    
#     numerator = dot(g_next, y)
#     denominator = dot(u, y)

#     return numerator/denominator
# end

# This is a modified Hestenses-Stiefel that restarts if ascent search direction is encountered.
struct SallehAlhawarat <: CGβConfig end

# sec 2 of (Salleh 2016)
function getβ(
    β_config::SallehAlhawarat,
    g_next::Vector{T},
    g::Vector{T},
    u::Vector{T},
    )::T where T

    norm_sq = norm(g_next)^2
    tmp = dot(g_next, g)

    if norm_sq > tmp
        numerator = norm_sq - tmp
        denominator = dot(u, g_next) - dot(u, g)

        return numerator/denominator
    end

    return zero(T)
end


struct LiuStorrey <: CGβConfig end

# sec 1 from (Yuan 2021).
function getβ(
    β_config::LiuStorrey,
    g_next::Vector{T},
    g::Vector{T},
    u::Vector{T},
    )::T where T

    y = g_next - g
    
    numerator = dot(g_next, y)
    denominator = -dot(u, y)

    return numerator/denominator
end


# from (Yuan 2021): future work.
#struct PolakRibièrePolyak end
