
# Algorithm 3.5 from (Nocedal 2006), strong wolfe. Using bisection.
struct LinesearchNocedal{T}
    #a_initial::T
    #const a_max::T
    c1::T # σ > 0
    c2::T # s > 0 
    a_max_growth_factor::T
    max_iters::Int
    zoom_max_iters::Int
end

# for c1, c2: see eqn 3.6-7 of Noceldal and eqns 8-9 of Yuan 20019.
function setupLinesearchNocedal(
    c1::T,
    c2::T;
    a_max_growth_factor::T = convert(T, 2),
    max_iters::Int = convert(Int, 1000),
    zoom_max_iters::Int = convert(Int, 100),
    )::LinesearchNocedal{T} where T

    @assert zero(T) < c1 < c2 < one(T)
    #@assert a_initial > zero(T)
    @assert max_iters >= 0
    @assert zoom_max_iters >= 0
    @assert a_max_growth_factor > 1

    #return LinesearchNocedal(a_initial, c1, c2, max_iters)
    return LinesearchNocedal(c1, c2, a_max_growth_factor, max_iters, zoom_max_iters)
end

# search via bisection. Algorithm 3.6 from (Nocedal 2006).
function linesearch!(
    info::LineSearchContainer{T},
    config::LinesearchNocedal{T},
    fdf!,
    f_x::T,
    df_x::Vector{T},
    a_initial::T,
    ) where T <: AbstractFloat

    # parse, checks.
    max_iters, zoom_max_iters = config.max_iters, config.zoom_max_iters
    c1, c2 = config.c1, config.c2
    a_max_growth_factor = config.a_max_growth_factor
    xp, df_xp, x, u = info.xp, info.df_xp, info.x, info.u

    if !( zero(T) < a_initial && isfinite(a_initial) )
        # invalid a_initial. Use default value.
        a_initial = one(T)
    end

    # set up.
    ϕ_0 = f_x
    dϕ_0 = dot(df_x, u)
    if dϕ_0 > zero(T)
        println("non-descent direction. Exit.")
        return ϕ_0, zero(T), 0, :non_descent_direction
    end

    a_prev = zero(T)
    ϕ_a_prev = ϕ_0

    a = a_initial
    a_max = a*a_max_growth_factor # do not supply a constant maximum.

    fdf_evals_ran = 0

    non_initial_iter = false
    for _ = 1:max_iters
        
        ϕ_a, dϕ_a = evalϕdϕ!(xp, df_xp, fdf!, a, x, u)
        fdf_evals_ran += 1

        chk1 = ϕ_a > ϕ_0 + c1*a*dϕ_0
        chk2 = ϕ_a >= ϕ_a_prev
        if chk1 || (chk2 && non_initial_iter)

            # zoom(a_prev, a)
            a_lb = a_prev
            a_ub = a
            ϕ_a_lb = ϕ_a_prev
            return zoom!(
                xp,
                df_xp,
                fdf!,
                x,
                u,
                a_lb,
                a_ub,
                ϕ_a_lb,
                ϕ_0,
                dϕ_0,
                c1,
                c2,
                fdf_evals_ran,
                zoom_max_iters,
            )
        end

        if abs(dϕ_a) <= -c2*dϕ_0

            return ϕ_a, a, fdf_evals_ran, :success
        end
        
        if dϕ_a >= 0

            # zoom(a, a_prev)
            a_lb = a
            a_ub = a_prev
            ϕ_a_lb = ϕ_a
            return zoom!(
                xp,
                df_xp,
                fdf!,
                x,
                u,
                a_lb,
                a_ub,
                ϕ_a_lb,
                ϕ_0,
                dϕ_0,
                c1,
                c2,
                fdf_evals_ran,
                zoom_max_iters,
            )
        end

        # ## pre-update: book keep
        a_prev = a
        ϕ_a_prev = ϕ_a
        non_initial_iter = true

        # ## update:choose a in [a, a_max]
        # use bisection for next candidate.
        a_max = a*a_max_growth_factor
        if a > a_max # a crude overflow detection.
            if verbose
                println("Overflow detected for linesearch upperbound update. Exit.")
            end
            return ϕ_a, a, fdf_evals_ran, :linesearch_a_max_overflow
        end
        a = (a_max + a)/2 # choose a via bisection.
    end

    if verbose
        println("Reached linesearch maximum iterations. Exit.")
    end

    return ϕ_a, a, fdf_evals_ran, :linesearch_max_iters_reached
end


# Alg 3.6 from (Nocedal 2006).
function zoom!(
    xp::Vector{T},
    df_xp::Vector{T},
    fdf!,
    x::Vector{T},
    u::Vector{T},
    a_lb::T,
    a_ub::T,
    ϕ_a_lb::T,
    ϕ_0::T,
    dϕ_0::T,
    c1::T,
    c2::T,
    fdf_evals_ran::Int,
    max_iters::Int
    ) where T

    # pre-allocate.
    a = zero(T)
    ϕ_a = zero(T)
    dϕ_a = zero(T)

    for _ = 1:max_iters

        # interpolate via bisection.
        a = (a_lb+a_ub)/2

        # zoom algorithm, inner loop.
        ϕ_a, dϕ_a = evalϕdϕ!(xp, df_xp, fdf!, a, x, u)
        fdf_evals_ran += 1

        if (ϕ_a > ϕ_0 + c1*a*dϕ_0) || (ϕ_a >= ϕ_a_lb)
            a_ub = a
        else
            if abs(dϕ_a) <= -c2*dϕ_0
                return ϕ_a, a, fdf_evals_ran, :success
            end

            if dϕ_a*(a_ub-a_lb) >= 0
                a_ub = a_lb
            end
            a_lb = a
            ϕ_a_lb = ϕ_a
        end
    end

    return ϕ_a, a, fdf_evals_ran, :zoom_max_iters_reached
end


###### Alg 2.2 in (Yuan 2019).


function minimizeobjective(
    fdf!,
    x_initial::Vector{T},
    config::CGConfig{T,ET},
    linesearch_config::LinesearchNocedal{T},
    ) where {T <: AbstractFloat, ET}

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
    a_initial = NaN # use default value on first try. Use HagerZhang 851 algorithm later.

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
        f_xp, a_star, fdf_evals_ran, status_symbol = linesearch!(
            info,
            linesearch_config,
            fdf!,
            f_x,
            df_x,
            a_initial,
        )
        a_initial = a_star
        if status_symbol != :success
            # return the last known good iterate.
            updateresult!(
                ret,
                x,
                df_x,
                f_x,
                n-1,
                status_symbol,
            )
            return ret
        end

        # step 4 & 5 (Yuan 2019): update iterate and objective-related evaluations.
        #f_x = fdf!(info.df_xp, x) # temporarily use info.df_xp to store the gradient of the next iterate.
        β = getβ(
            config.β_config,
            info.df_xp,
            df_x,
            info.u,            
        )
        x[:] = info.xp
        f_x = f_xp
        df_x[:] = info.df_xp # now, safe to overwrite gradient of the iterate.
        info.x[:] = x
        norm_df_x = norm(df_x)

        # step 5: update search direction for next iteration.
        updatedir!(info.u, df_x, β)

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
