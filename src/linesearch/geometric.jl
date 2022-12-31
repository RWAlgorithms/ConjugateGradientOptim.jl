############ Inexact linesearch for the various Armijo-based conditions.

abstract type GeometricStepTrait end
struct DivideGeometricStep <: GeometricStepTrait end
struct MultiplyGeometricStep <: GeometricStepTrait end

function getgeometricstep(::DivideGeometricStep, a::T, ρ::T)::T where T
    return a/ρ
end

function getgeometricstep(::MultiplyGeometricStep, a::T, ρ::T)::T where T
    return a*ρ
end

struct Backtracking{T,CT} <: LineSearchConfig
    condition::CT
    discount_factor::T
    max_iters::Int
    feasibility_max_iters::Int
end

function linesearch!(
    info::LineSearchContainer{T},
    config::Backtracking{T,CT},
    fdf!,
    f_x::T,
    df_x::Vector{T},
    a_initial::T,
    )::Tuple{T,T,Int,Symbol} where {T <: AbstractFloat, CT}

    # parse.
    xp, df_xp, x, u = info.xp, info.df_xp, info.x, info.u
    discount_factor, max_iters = config.discount_factor, config.max_iters
    condition = config.condition
    feasibility_max_iters = config.feasibility_max_iters

    # setup
    ϕ_0 = f_x
    if !isfinite(ϕ_0)
        return ϕ_0, zero(T), 0, :accepted_non_finite_iterate
    end

    dϕ_0 = dot(df_x, u)
    if dϕ_0 > zero(T)
        return ϕ_0, zero(T), 0, :non_descent_search_direction
    end

    fdf_evals_ran = 0
    a = a_initial
    if !isfinite(a)
        # use default 1.
        a = abs(ϕ_0)/dot(u,u)
    end
    if !isfinite(a)
        # use default 2.
        a = one(T)
    end
    # hard code for now.
    reduction_factor = convert(T, 0.5)
    ϕ_a, dϕ_a, a, fdf_evals_ran, status_flag = findfeasiblestepsize!(
        df_xp,
        xp,
        fdf!,
        fdf_evals_ran,
        a,
        x,
        u,
        reduction_factor,
        zero(T);
        max_iters = feasibility_max_iters,
    )
    if status_flag != :feasible
        #@show ϕ_0, dϕ_0
        return ϕ_0, zero(T), 0, :cannot_find_initial_feasible_step
    end

    # evaluate.
    ϕ_a, _ = evalϕdϕ!(xp, df_xp, fdf!, a, x, u)
    fdf_evals_ran += 1

    valid_step = evalbacktrackcondition(condition, ϕ_a, a, ϕ_0, dϕ_0)

    if valid_step
        # increase step.
        return geometricsearch!(
            xp, df_xp, fdf!, a, x, u, condition, max_iters, discount_factor,
            DivideGeometricStep(),
            fdf_evals_ran, ϕ_a, ϕ_0, dϕ_0,
        )
    else
        # decrease step.
        return geometricsearch!(
            xp, df_xp, fdf!, a, x, u, condition, max_iters, discount_factor,
            MultiplyGeometricStep(),
            fdf_evals_ran, ϕ_a, ϕ_0, dϕ_0,
        )
    end

    return ϕ_a, a, fdf_evals_ran, :linesearch_max_iters_reached
end

function geometricsearch!(
    xp::Vector{T},
    df_xp::Vector{T},
    fdf!,
    a::T,
    x::Vector{T},
    u::Vector{T},
    condition::CT,
    max_iters,
    discount_factor::T,
    geta_method::GeometricStepTrait,
    fdf_evals_ran::Int,
    ϕ_a::T,
    ϕ_0::T,
    dϕ_0::T,
    ) where {T,CT}
    
    #a = clamp(a, zero(T), convert(T, Inf))
    #@show a, ϕ_a, ϕ_0, dϕ_0
    # assume ϕ_0, dϕ_0 are finite. i.e. df_x, u, f_x are per-element finite.

    a_prev::T = a
    ϕ_a_prev::T = ϕ_a
    for _ = 1:max_iters
    
        a = getgeometricstep(geta_method, a, discount_factor)
        if !isfinite(a) # if increasing up to overflow.
            return ϕ_a_prev, a_prev, fdf_evals_ran, :non_finite_step_proposed
        end

        if a == a_prev # if decreasing down to zero.
            return ϕ_a_prev, a_prev, fdf_evals_ran, :proposed_step_same_as_current_step
        end

        # evaluate.
        ϕ_a, _ = evalϕdϕ!(xp, df_xp, fdf!, a, x, u)
        fdf_evals_ran += 1

        valid_step = evalbacktrackcondition(condition, ϕ_a, a, ϕ_0, dϕ_0)
        if !valid_step
            #@show ϕ_a_prev, a_prev, ϕ_a, a
            return ϕ_a_prev, a_prev, fdf_evals_ran, :success
        end

        # book keep.
        a_prev = a
        ϕ_a_prev = ϕ_a
    end

    return ϕ_a, a, fdf_evals_ran, :linesearch_max_iters_reached
end



########### conditions

# eqn 8 in (Shi 2005)
struct Armijo{T}
    c1::T # this is σ in eqn 8 of (Shi 2005).
    #L::T
end

function evalbacktrackcondition(
    condition::Armijo{T},
    ϕ_a::T,
    a::T,
    ϕ_0::T, # f_x
    dϕ_0::T, # dot(df_x, u)
    )::Bool where T

    # parse.
    c1 = condition.c1
    @assert zero(T) < c1 < one(T)
    
    # check feasibility.
    if !isfinite(ϕ_0) || !isfinite(ϕ_a) || !isfinite(a)
        return false
    end

    # the condition for deciding whether the step size is too large.
    LHS1 = ϕ_0 - ϕ_a
    chk1 = LHS1 >= -c1*a*dϕ_0

    return chk1
end