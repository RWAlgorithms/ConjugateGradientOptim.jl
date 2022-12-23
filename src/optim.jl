
# Algorithm 3.5 from (Nocedal 2006), strong wolfe. Using bisection.
mutable struct LinesearchNocedal{T}
    a_initial::T
    #const a_max::T
    const c1::T # σ > 0
    const c2::T # s > 0 
    const max_iters::Int
end

function setupLinesearchNocedal(
    a_initial::T;
    c1 = convert(T, 1e-5), # δ.  # see eqn 3.6-7 of Noceldal and eqns 8-9 of Yuan 20019.
    c2 = convert(T, 0.8), # σ
    max_iters = 1000,
    ) where T

    @assert zero(T) < c1 < c2 < one(T)
    @assert a_initial > zero(T)
    @assert max_iters >= 0

    return LinesearchNocedal(a_initial, c1, c2, max_iters)
end

# search via bisection. Algorithm 3.6 from (Nocedal 2006).
function linesearch!(
    info::LineSearchContainer{T},
    config::LinesearchNocedal{T},
    fdf!,
    f_x::T,
    df_x::Vector{T},
    ) where T <: AbstractFloat

    max_iters = config.max_iters
    c1, c2, a1 = config.c1, config.c2, config.a1
    xp, df_xp, x, u = info.xp, info.df_xp, info.x, info.u

    #norm_u_sq = dot(u,u)
    ϕ_0 = f_x
    dϕ_0 = dot(df_x,u)
    if dϕ_0 < zero(T)
        println("non-descent direction. Exit.")
        return convert(T, NaN), convert(T, NaN), 0 #  TODO.
    end

    a_prev = zero(T)
    ϕ_a_prev = ϕ_0
    dϕ_a_prev = dϕ_0

    a = a1
    a_max = a*10 # do not supply a constant maximum.

    non_initial_iter = false
    for i = 1:max_iters
        
        ϕ_a, dϕ_a = evalϕdϕ!(xp, df_xp, fdf!, a, x, u)

        chk1 = ϕ_a > ϕ_0 + c1*a*dϕ_0
        chk2 = ϕ_a >= ϕ_a_prev
        if chk1 || (chk2 && non_initial_iter)
            a = zoom(a_prev, a)
            return a
        end

        if abs(dϕ_a) <= -c2*dϕ_0
            return a
        end
        
        if dϕ_a >= 0
            a = zoom(a, a_prev)
            return a
        end

        # ## pre-update: book keep
        a_prev = a
        ϕ_a_prev = ϕ_a
        dϕ_a_prev = dϕ_a
        non_initial_iter = true

        # ## update:choose a in [a, a_max]
        # use bisection for next candidate.
        a = (a_max + a)/2
    end

    if verbose
        println("Linesearch maximum hit. Terminate.")
    end

    return xp, convert(T,NaN), convert(T,NaN), i
end



###### Alg 2.2 in (Yuan 2019).


function minimizeobjective(
    fdf!,
    x_initial::Vector{T},
    config::CGConfig{T,ET},
    linesearch_config::LinesearchNocedal{T},
    ) where {T <: AbstractFloat, ET}

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
    linesearch_iters_ran::Int = -1
    
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
        f_xp, norm_df_xp, a_star, linesearch_iters_ran, success_flag = linesearch!(
            info,
            linesearch_config,
            fdf!,
            f_x,
            df_x,
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
                linesearch_iters_ran,
                n,
            )

            return ret
        end

        # update the iterate `x` using the linesearch solution in `info`.
        updateiterateoptim!(
            x,
            info.df_xp,
            norm_df_xp,
            a_star,
            info.u,
        )

        # update objective-related evaluations using `x`, and update `β`.
        f_x = fdf!(info.df_xp, x) # temporarily use info.df_xp to store the gradient of the next iterate.
        β = getβ(
            info.df_xp,
            df_x,
            info.u,
            config.μ,
        )
        df_x[:] = info.df_xp # now, safe to overwrite gradient of the iterate.
        info.x[:] = x
        norm_df_x = norm(df_x)

        # step 2 (Yuan 2019): update search direction u.
        updatedir!(info.u, df_x, β)

        # update trace.
        updatetrace!(
            ret.trace,
            f_x,
            norm_df_x,
            linesearch_iters_ran,
            n,
        )
    end

    updateresult!(
        ret,
        x,
        df_x,
        f_x,
        max_iters,
        :max_iters,
    )
    return ret
end

# `xp` is z in Alg 3.1.
function updateiterateoptim!(
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