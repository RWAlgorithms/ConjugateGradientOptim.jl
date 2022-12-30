

########### generic core routines.

function updatedir!(
    u::Vector{T},
    df_x::Vector{T},
    B::Matrix{T},
    ) where T <: AbstractFloat

    @assert length(u) == length(df_x) == size(B,1) == size(B,2)

    if !isposdef(B)
        B[:] = diagm(ones(T,size(B,1))) #df_x*df_x'
        
        # debug.
        println("reset B")
    end
    u[:] = B\(-df_x)

    return nothing
end

# quasi-Newton.
function initializeLineSearchContainer!(
    info::LineSearchContainer{T},
    β_config::QNβConfig,
    df_x::Vector{T},
    x::Vector{T},
    ) where T
    
    B = β_config.B
    if !isposdef(B)
        # invalid initial matrix. use default: rank-1 matrix from gradient.
        B[:] = diagm(ones(T,size(B,1))) #df_x*df_x'
    end

    info.u[:] = B\(-df_x) # search direction. eqn 3.2 in (Yuan 2017).
    info.x[:] = x
    info.xp[:] = x
    info.df_xp[:] = df_x
    
    return nothing
end

function initializeβ(::Type{T}, β_config::QNβConfig) where T
    return β_config.B
end

########### quasi-Newton variants.

# (Yuang 2019): modified Hager-Zhang, to have trust region behavior.
struct BroydenFamily{T} <: QNβConfig
    θ::T
    B::Matrix{T}
end

function setupBroydenFamily(θ::T, N::Int)::BroydenFamily{T} where T <: AbstractFloat

    @assert zero(T) <= θ #<= one(T)
    B = Matrix{T}(undef,N,N)
    fill!(B, convert(T, NaN)) # this indicates algorithm to use the default for initial B.

    return BroydenFamily(θ, B)
end

# sec 1 of (Yuan 2017). This is getting the B quantity.
function getβ(
    B_config::BroydenFamily{T},
    g_next::Vector{T},
    g::Vector{T},
    u::Vector{T},
    )::Matrix{T} where T

    # parse, set up.
    θ, B = B_config.θ, B_config.B
    
    y = g_next - g
    
    # quasi-Newton equation. eqn 1.4.
    s = B\y

    Bs = B*s
    sBs = dot(s,Bs)
    tmp = θ*sBs
    v = y ./ dot(s,y) .- Bs ./ sBs
    B[:] = B .- Bs*Bs' ./sBs .+ y*y' ./dot(s,y) + tmp .* v*v'

    return B
end