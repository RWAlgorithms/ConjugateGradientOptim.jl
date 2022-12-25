
#### various types of conjugate gradient β values.
# TODO: make these updates non-allocating.


# from (Yuang 2019):
struct YuanHagerZhang{T} <: βConfig
    μ::T
end

function setupYuanHagerZhang(μ::T)::YuanHagerZhang{T} where T

    @assert zero(T) < μ < one(T)

    return YuanHagerZhang(μ)
end

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



struct HagerZhang <: βConfig end


# sec 2.2 of (Yuan 2019).
# This is the YuanHagerZhang version, but set R = R2.
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

# TODO: HS: https://www.sciencedirect.com/science/article/pii/S1877705811017772?ref=cra_js_challenge&fr=njs
struct HestensesStiefel <: βConfig end

# sec 1 from (Yuan 2021).
function getβ(
    β_config::HestensesStiefel,
    g_next::Vector{T},
    g::Vector{T},
    u::Vector{T},
    )::T where T

    y = g_next - g
    
    numerator = dot(g_next, y)
    denominator = dot(u, y)

    return numerator/denominator
end


struct LiuStorrey <: βConfig end

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

#struct LS <: βConfig end

# from (Yuan 2021): Polak–Ribière–Polyak
#struct PolakRibièrePolyak end