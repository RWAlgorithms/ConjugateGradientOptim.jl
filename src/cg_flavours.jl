
#### various types of conjugate gradient β values.



# from (Yuang 2019):
struct YuanHagerZhang{T} <: βConfig
    μ::T
end

function setupYuanHagerZhang(μ::T)::YuanHagerZhang{T} where T

    @assert zero(T) < μ < one(T)

    return YuanHagerZhang(μ)
end

struct HagerZhang <: βConfig end

# from (Yuan 2017):
#struct PRP end

# sec 2.2 of (Yuan 2019).
function getβ(
    β_config::YuanHagerZhang{T},
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