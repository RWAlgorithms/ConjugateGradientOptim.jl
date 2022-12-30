

########### quasi-newton.

# (Yuang 2019): modified Hager-Zhang, to have trust region behavior.
struct BroydenFamily{T} <: QNβConfig
    θ::T
    B::Matrix{T}
end

function setupBroydenFamily(θ::T, N::Int)::BroydenFamily{T} where T

    @assert zero(T) <= θ <= one(T)

    return BroydenFamily(θ, Matrix{T}(undef,N,N))
end

# sec 1 of (Yuan 2017).
function getβ(
    β_config::BroydenFamily{T},
    g_next::Vector{T},
    g::Vector{T},
    u::Vector{T},
    )::T where T

    # parse, set up.
    θ, B = β_config.θ, β_config.B
    
    y = g_next - g
    
    # quasi-Newton equation. eqn 1.4.
    s = B\y

    Bs = B*s
    sBs = dot(s,Bs)
    tmp = θ*sBs
    v = y ./ dot(s,y) .- Bs ./ sBs
    B[:] = B .- Bs*Bs' ./sBs .+ y*y' ./dot(s,y) + tmp .* v*v'

    # solve for direction.
    β_MN = dot(tmp1, tmp2)

    return β_MN
end