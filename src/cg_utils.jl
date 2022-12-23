
function updatedir!(
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


# sec 2.2 of (Yuan 2019).
function getβ(
    g_next::Vector{T},
    g::Vector{T},
    u::Vector{T},
    μ::T, # 0 < μ < 1
    )::T where T

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