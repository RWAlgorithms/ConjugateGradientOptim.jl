
# Algorithm 3.5 from (Nocedal 2006), strong wolfe. Using bisection.
struct StrongWolfeBisection{T} <: LineSearchConfig
    #a_initial::T
    #const a_max::T
    c1::T # σ > 0
    c2::T # s > 0 
    a_max_growth_factor::T
    max_iters::Int
    zoom_max_iters::Int
end

# for c1, c2: see eqn 3.6-7 of Noceldal and eqns 8-9 of Yuan 20019.
function setupStrongWolfeBisection(
    c1::T,
    c2::T;
    a_max_growth_factor::T = convert(T, 2),
    max_iters::Int = convert(Int, 1000),
    zoom_max_iters::Int = convert(Int, 100),
    )::StrongWolfeBisection{T} where T

    @assert zero(T) < c1 < c2 < one(T)
    #@assert a_initial > zero(T)
    @assert max_iters >= 0
    @assert zoom_max_iters >= 0
    @assert a_max_growth_factor > 1

    #return StrongWolfeBisection(a_initial, c1, c2, max_iters)
    return StrongWolfeBisection(c1, c2, a_max_growth_factor, max_iters, zoom_max_iters)
end

# search via bisection. Algorithm 3.6 from (Nocedal 2006).
function linesearch!(
    info::LineSearchContainer{T},
    config::StrongWolfeBisection{T},
    fdf!,
    f_x::T,
    df_x::Vector{T},
    a_initial::T;
    verbose = false,
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

        if verbose
            println("non-descent direction. Exit.")
        end
        return ϕ_0, zero(T), 0, :non_descent_search_direction
    end

    a_prev = zero(T)
    ϕ_a_prev = ϕ_0

    a = a_initial
    ϕ_a = ϕ_0
    dϕ_a = dϕ_0
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
