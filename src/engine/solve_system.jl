# (Yuan 2019):
# Yuan, G., Wang, B., & Sheng, Z. (2019). The Hager–Zhang conjugate gradient algorithm for large-scale nonlinear equations. International Journal of Computer Mathematics, 96(8), 1533-1547.

############# line search.

# parameters in eqn 18 of (Yuan 2019).
struct LinesearchSolveSys{T}
    ρ::T # 0 < ρ < 1
    σ::T # σ > 0
    s::T # s > 0 
    max_iters::Int
end

function setupLinesearchSolveSys(
    s::T;
    σ = convert(T, 0.5),
    ρ = convert(T, 0.95),
    max_iters = round(Int, log(ρ, 1e-6)),
    ) where T

    @assert zero(T) < ρ < one(T)
    @assert ρ > zero(T)
    @assert s > zero(T)

    return LinesearchSolveSys(ρ, σ, s, max_iters)
end

# search from pre-specified bracket.
function linesearch!(
    info::LineSearchContainer{T},
    config::LinesearchSolveSys{T},
    fdf!,
    ) where T <: AbstractFloat

    max_iters = config.max_iters
    ρ, σ, a0 = config.ρ, config.σ, config.s
    xp, df_xp, x, u = info.xp, info.df_xp, info.x, info.u

    norm_u_sq = dot(u,u)

    for i = 0:max_iters-1
        a = a0*ρ^i
        
        f_xp, dϕ_xp = evalϕdϕ!(xp, df_xp, fdf!, a, x, u)

        # weeak Wolfe.
        norm_df_xp = norm(df_xp)
        if !( -dϕ_xp < σ*a*norm_df_xp*norm_u_sq )

            return f_xp, norm_df_xp, a, i, true
        end
    end

    return xp, convert(T,NaN), convert(T,NaN), i, false
end

################# cg algorithm for solving a non-linear system. g(x) = 0, g: R^D to R^D.
# although f is computed, does not care about ascent or descent directions.
# in this case, f := 0.5*norm(g(x))^2, which means df_x = g(x)? still useful to keep track of f(x)'s changes for visualizing the trace.
#   -therefore, we're not removing the referenes to f_x in solvesystem().

# allows for objective to increase.
# use `u` here as `d` from (Yuan 2019).
function solvesystem(
    fdf!,
    x_initial::Vector{T},
    config::CGConfig{T,BT,ET},
    linesearch_config::LinesearchSolveSys{T},
    ) where {T <: AbstractFloat, BT, ET}

    # # Set up.

    # ## parse.
    D = length(x_initial)
    max_iters = config.max_iters
    #linesearch_config = config.linesearch_config

    # ## allocate.
    df_x = Vector{T}(undef, D)
    x = copy(x_initial)
    #u = Vector{T}(undef, D) 

    # ## Step 1 (Yuan 2019): initialize or allocate for first iteration.
    f_x::T = fdf!(df_x, x)
    norm_df_x::T = norm(df_x)
    β::T = zero(T)
    norm_df_xp::T = convert(T, NaN)
    fdf_evals_ran::Int = -1
    
    # ## return container.
    ret = Results(
        f_x,
        x,
        df_x,
        0,
        :incomplete,
        setuptrace(T, config.trace_status),
    )
    resizetrace!(ret.trace, max_iters)

    # ## line search.
    info = LineSearchContainer(T, D)
    info.u[:] = -df_x # search direction.
    info.x[:] = x
    info.xp[:] = x
    info.df_xp[:] = df_x

    # # Run algortihm.
    for n = 1:max_iters

        # check stopping condition.
        if norm_df_x < config.ϵ
            # the previous iteration meets the stopping criteria. Return it.
            updateresult!(
                ret,
                x,
                df_x,
                f_x,
                n-1,
                :success,
            )
            return ret
        end

        # step 3 (Yuan 2019): linesearch.
        f_xp, norm_df_xp, a_star, fdf_evals_ran, success_flag = linesearch!(
            info,
            linesearch_config,
            fdf!,
        )
        if !success_flag
            # return the last known good iterate.
            updateresult!(
                ret,
                x,
                df_x,
                f_x,
                n-1,
                :linesearch_failed,
            )
            return ret
        end

        # step 4 & 5 (Yuan 2019): update iterate and objective-related evaluations.

        if norm_df_xp < config.ϵ
            # return the linesearch solution.
            updateresult!(
                ret,
                info.xp,
                info.df_xp,
                f_xp,
                n,
                :success,
            )

            # early exit, force update last trace.
            updatetrace!(
                ret.trace,
                f_xp,
                norm(info.df_xp),
                a_star,
                fdf_evals_ran,
                n,
            )

            return ret
        end

        # update the iterate `x` using the linesearch solution in `info`.
        updateiteratesolvesys!(
            x,
            info.df_xp,
            norm_df_xp,
            a_star,
            info.u,
        )

        # update objective-related evaluations using `x`, and update `β`.
        f_x = fdf!(info.df_xp, x) # temporarily use info.df_xp to store the gradient of the next iterate.
        β = getβ(
            config.β_config,
            info.df_xp,
            df_x,
            info.u,
        )
        df_x[:] = info.df_xp # now, safe to overwrite gradient of the iterate.
        info.x[:] = x
        norm_df_x = norm(df_x)

        # step 2 (Yuan 2019): update search direction u.
        updatedir!(config.β_config, info.u, df_x, β)

        # update trace.
        updatetrace!(
            ret.trace,
            f_x,
            norm_df_x,
            a_star,
            fdf_evals_ran,
            n,
        )
    end

    updateresult!(
        ret,
        x,
        df_x,
        f_x,
        max_iters,
        :max_iters_reached,
    )
    return ret
end

# `xp` is z in Alg 3.1.
function updateiteratesolvesys!(
    x::Vector{T},
    df_xp::Vector{T},
    norm_df_xp::T,
    a::T,
    u::Vector{T},
    ) where T

    #m = a*dot(df_xp, u)/dot(df_x , df_x)
    m = a*dot(df_xp, u)/norm_df_xp^2 # omit negative sign so that (1) has an addition.
    
    for i in eachindex(x)
        x[i] = x[i] + m*df_xp[i] # (1).
    end

    return nothing
end


