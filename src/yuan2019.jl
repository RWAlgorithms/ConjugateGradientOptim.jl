# (Yuan 2019):
# Yuan, G., Wang, B., & Sheng, Z. (2019). The Hager–Zhang conjugate gradient algorithm for large-scale nonlinear equations. International Journal of Computer Mathematics, 96(8), 1533-1547.



struct Yuan2019{T}
    ρ::T # 0 < ρ < 1
    σ::T # σ > 0
    s::T # s > 0 
    ϵ::T # 0 < ϵ < 1
    η::T # 0 < η < 1 # this is μ in the paper.
    linesearch_max_iters::Int
    max_iters::Int
end

mutable struct Diagnostics{T}
    min_cost::T
    minimizer::Vector{T}
    grad::Vector{T}
    iters_ran::Int
    status::Symbol
end

function updatediagnostics!(
    ret::Diagnostics{T},
    info::LineSearchContainer{T},
    f_x::T,
    i::Int,
    status::Symbol) where T

    ret.min_cost = f_x
    ret.minimizer[:] = info.xp
    ret.grad[:] = info.df_xp
    ret.iters_ran = i
    ret.status = status

    return nothing
end

function updatediagnostics!(
    ret::Diagnostics{T},
    x::Vector{T},
    df_x::Vector{T},
    f_x::T,
    i::Int,
    status::Symbol) where T

    ret.min_cost = f_x
    ret.minimizer[:] = x
    ret.grad[:] = df_x
    ret.iters_ran = i
    ret.status = status

    return nothing
end

# use `u` here as `d` from (Yuan 2019).
function runcgyuan2019(
    fdf!,
    x_initial::Vector{T},
    config::Yuan2019{T},
    ) where T <: AbstractFloat

    # allocate.
    D = length(x_initial)
    
    df_x = Vector{T}(undef, D)
    x = copy(x_initial)
    #u = Vector{T}(undef, D) 

    # setup.
    f_x = fdf!(df_x, x)
    ret = Diagnostics(
        f_x,
        x,
        df_x,
        0,
        :incomplete,
    )

    info = LineSearchContainer(T, D)
    info.u[:] = -df_x # search direction.
    info.x[:] = x
    info.xp[:] = x
    info.df_xp[:] = df_x

    norm_df_x = norm(df_x)
    β = zero(T)
    norm_df_xp = convert(T, NaN)

    # trace
    f_x_trace = Vector{T}(undef, config.max_iters)
    df_x_norm_trace = Vector{T}(undef, config.max_iters)

    # run algortihm.
    for n = 1:config.max_iters

        # check stopping condition.
        if norm_df_x < config.ϵ
            updatediagnostics!(
                ret,
                x,
                df_x,
                f_x,
                n,
                :success,
            )
            return ret
        end

        # step3: linesearch.
        f_xp, norm_df_xp, a_star, success_flag = linesearchyuan2019!(
            info,
            fdf!,
            config.ρ,
            config.s,
            config.σ;
            max_iters = config.linesearch_max_iters
        )
        if !success_flag
            # return the last known good iterate.
            updatediagnostics!(
                ret,
                x,
                df_x,
                f_x,
                n,
                :linesearch_failed,
            )
            return ret
        end

        # step 5: update iterate and cost-related evaluations.

        if norm_df_xp < config.ϵ
            # return the linesearch solution.
            updatediagnostics!(
                ret,
                info.xp,
                info.df_xp,
                f_xp,
                n,
                :success,
            )

            return ret
        end

        # update the iterate `x` using the linesearch solution in `info`.
        updateiterate!(
            x,
            info.df_xp,
            norm_df_xp,
            a_star,
            info.u,
        )

        # update cost-related evaluations using `x`, and update `β`.
        f_x = fdf!(info.df_xp, x) # temporarily use info.df_xp to store the gradient of the next iterate.
        β = getβyuan2019(
            info.df_xp,
            df_x,
            info.u,
            config.η,
        )
        df_x[:] = info.df_xp # now, safe to overwrite gradient of the iterate.
        info.x[:] = x

        # store trace
        f_x_trace[n] = f_x

        # step 2: update search direction u.
        updatedir!(info.u, df_x, β)

    end

    updatediagnostics!(
        ret,
        x,
        df_x,
        f_x,
        config.max_iters,
        :max_iters,
    )
    return ret, f_x_trace
end

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

function linesearchyuan2019!(
    info::LineSearchContainer{T},
    fdf!,
    ρ::T,
    a0::T,
    σ::T;
    max_iters::Int = 1000,
    verbose::Bool = false,
    ) where {FT,T}

    xp, df_xp, x, u = info.xp, info.df_xp, info.x, info.u

    norm_u_sq = dot(u,u)

    for i = 0:max_iters-1
        a = a0*ρ^i
        
        f_xp, dϕ_xp = evalϕdϕ!(xp, df_xp, fdf!, a, x, u)

        # weeak Wolfe.
        norm_df_xp = norm(df_xp)
        if !( -dϕ_xp < σ*a*norm_df_xp*norm_u_sq )

            return f_xp, norm_df_xp, a, true
        end
    end

    if verbose
        println("Linesearch maximum hit. Terminate.")
    end

    return xp, convert(T,NaN), convert(T,NaN), false
end

# `xp` is z in Alg 3.1.
function updateiterate!(
    x::Vector{T},
    df_xp::Vector{T},
    norm_df_xp::T,
    a::T,
    u::Vector{T},
    ) where T

    #m = a*dot(df_xp, u)/dot(df_x , df_x)
    m = a*dot(df_xp, u)/norm_df_xp^2
    
    for i in eachindex(x)
        x[i] = x[i] - m*df_xp[i]
    end

    return nothing
end

# sec 2.2 of (Yuan 2019).
function getβyuan2019(
    g_next::Vector{T},
    g::Vector{T},
    u::Vector{T},
    η::T, # 0 < η < 1
    )::T where T

    y = g_next - g
    
    R1 = η*norm(u)*norm(y)
    R2 = dot(u,y)
    R3 = 2*dot(y,y)*dot(u,g_next)/dot(y,g_next)
    R = max(R1, R2, R3)
    #R = R2

    tmp2 = g_next ./ R

    m = 2*dot(y,y)/R
    tmp1 = y - m .* u

    β_MN = dot(tmp1, tmp2)

    return β_MN
end