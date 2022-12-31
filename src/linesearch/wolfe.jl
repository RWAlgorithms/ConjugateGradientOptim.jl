
######### Inexact linesearch for the various Wolfe-based conditions.

# based on https://sites.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf

struct WolfeBisection{T,CT} <: LineSearchConfig
    condition::CT
    max_iters::Int
    max_step_size::T
    feasibility_max_iters::Int
end

function linesearch!(
    info::LineSearchContainer{T},
    config::WolfeBisection{T,CT},
    fdf!,
    f_x::T,
    df_x::Vector{T},
    a_initial::T,
    ) where {T <: AbstractFloat,CT}

    # hard code for now.
    reduction_factor = convert(T, 0.5)
    growth_factor = convert(T, 2)

    # parse.
    xp, df_xp, x, u = info.xp, info.df_xp, info.x, info.u
    condition, max_step_size = config.condition, config.max_step_size
    max_iters, feasibility_max_iters = config.max_iters, config.feasibility_max_iters
    if !(max_step_size > a_initial > zero(T))
        a_initial = min(one(T), convert(T,max_step_size/2)) # reset a_initial.
    end
    
    # set up. initialize.
    ϕ_0 = f_x
    if !isfinite(ϕ_0)
        return ϕ_0, zero(T), 0, :accepted_non_finite_iterate
    end

    dϕ_0 = dot(df_x, u)
    if dϕ_0 > zero(T)
        return ϕ_0, zero(T), 0, :non_descent_search_direction
    end
    
    a = a_initial
    fdf_evals_ran = 0
    lb = zero(T)
    ub = convert(T, Inf)

    # make sure the initial step size is feasible.
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
        return ϕ_0, zero(T), 0, :cannot_find_initial_feasible_step
    end

    for _ = 1:max_iters

        # get conditions.
        valid_large_step, valid_small_step = evalwolfeconditions(
            condition,
            ϕ_a,
            dϕ_a,
            a,
            u,
            ϕ_0,
            dϕ_0,
        )        

        if !valid_large_step || !valid_small_step
            if !valid_large_step
                
                #@show lb, ub, a, a_initial, norm(df_x)

                # adjust upper bound because proposed step is too large.
                ub = a
                
                # # reset.
                # if !(ub > lb)
                #     a = one(T)
                #     println("reset")
                # end

                # choose the midpoint, but decrease if infeasible.
                a = (lb+ub)/2
            else
                # adjust lower bound because proposed step is too small.
                lb = a
                if !isfinite(ub)
                    # the case we have never had an invalid step due to it being too large, so ub still at Inf.
                    a0 = a
                    a = growth_factor*a
                    #@show valid_large_step, valid_small_step, ϕ_a, dϕ_a, a, u, ϕ_0, dϕ_0
                    if a > max_step_size
                        #@show valid_large_step, valid_small_step, ϕ_a, dϕ_a, a, u, ϕ_0, dϕ_0
                        # chk1 = valid_large_step
                        # chk2 = valid_small_step
                        # @show chk1, chk2, ϕ_a, dϕ_a, a0, a, u, ϕ_0, dϕ_0
                        # println("reset a")
                        # a = one(T)
                        return ϕ_0, zero(T), 0, :max_step_length_reached
                    end
                else
                    a = (lb+ub)/2
                end
            end

            # chk1 = valid_large_step
            # chk2 = valid_small_step
            # @show chk1, chk2, a, a_initial, norm(df_x)
            
            if !(lb < a < ub)
                if !isapprox(norm(u+df_x), zero(T))
                    # reset.
                    lb = zero(T)
                    ub = convert(T,Inf)
                    #a = one(T)
                    a = a_initial
                    u[:] = -df_x
                else
                    ϕ_0, zero(T), 0, :step_bracket_precision_issue
                end
            end

            # decrease step length if infeasible.
            ϕ_a_back = ϕ_a
            dϕ_a_back = dϕ_a
            a_back = a
            fdf_evals_ran_back = fdf_evals_ran

            ϕ_a, dϕ_a, a, fdf_evals_ran, status_flag = findfeasiblestepsize!(
                df_xp,
                xp,
                fdf!,
                fdf_evals_ran,
                a,
                x,
                u,
                reduction_factor,
                lb;
                max_iters = feasibility_max_iters,
            )
            if status_flag != :feasible
                @show ϕ_a, dϕ_a, a, fdf_evals_ran_back, ϕ_a_back, dϕ_a_back, a_back, fdf_evals_ran
                @show lb, a, a_back
                @show valid_large_step, valid_small_step # got it! both false.
                return ϕ_0, zero(T), 0, :cannot_find_feasible_step
            end
        else
            return ϕ_a, a, fdf_evals_ran, :success
        end
    end

    return ϕ_a, a, fdf_evals_ran, :linesearch_max_iters_reached
end

# when f might return a non-finite value (e.g. barrier method centering step objectives or out of domain implementation).
# assumes a = 0 must be feasbiel.
# if the supplied a is infeasible, reduce a until feasible, until lb is reached.
# lb must be smaller than a.
function findfeasiblestepsize!(
    df_xp::Vector{T},
    xp::Vector{T},
    fdf!,
    fdf_evals_ran::Int,
    a::T,
    x::Vector{T},
    u::Vector{T},
    reduction_factor::T,
    lb::T;
    max_iters::Int = 300,
    )::Tuple{T,T,T,Int,Symbol} where T
    
    # checks.
    @assert zero(T) < reduction_factor < one(T)
    if lb > a
        return zero(T), zero(T), a, fdf_evals_ran, :bisection_lower_bound_larger_than_proposed_step
    end

    # evaluate.
    ϕ_a, dϕ_a = evalϕdϕ!(xp, df_xp, fdf!, a, x, u)
    fdf_evals_ran += 1

    iter = 1
    while a > lb && iter < max_iters
        if isfinite(ϕ_a) && isfinite(dϕ_a)
            return ϕ_a, dϕ_a, a, fdf_evals_ran, :feasible
        end

        a = a*reduction_factor
        ϕ_a, dϕ_a = evalϕdϕ!(xp, df_xp, fdf!, a, x, u)
        fdf_evals_ran += 1
        iter += 1
    end

    return ϕ_a, dϕ_a, a, fdf_evals_ran, :infeasible
end

######### conditions.

# modified Wolfe condition.
# equations 1.10, 1.11 from (Yuan 2017).
struct YuanWeiLuWolfe{T}
    c1::T # this is δ in eqn 1.8.
    c2::T # this is σ in eqn 1.9.
    δ1::T # this is δ1 in eqn 1.10, 1.11.
end

function evalwolfeconditions(
    condition::YuanWeiLuWolfe{T},
    ϕ_a::T,
    dϕ_a::T,
    a::T,
    u::Vector{T},
    ϕ_0::T, # f_x
    dϕ_0::T, # dot(df_x, u)
    )::Tuple{Bool,Bool} where T

    # parse.
    c1, c2, δ1 = condition.c1, condition.c2, condition.δ1

    # checks.
    @assert zero(T) < δ1 < c1 < c2 < one(T)

    # check these in the calling routine.
    #@assert dϕ_0 < zero(T) 
    #@assert isfinite(ϕ_0)
    #@assert isfinite(ϕ_a)

    norm_u_sq = dot(u,u)

    # the condition for deciding whether the step size is too large.
    RHS1 = ϕ_0 + c1*a*dϕ_0 + a*min( -δ1*dϕ_0, c1*a*norm_u_sq/2)
    chk1 = ϕ_a <= RHS1

    # the condition for deciding whether the step size is too small.
    RHS2 = c2*dϕ_0 + min( -δ1*dϕ_0, c1*a*norm_u_sq)
    chk2 = dϕ_a >= RHS2

    return chk1, chk2
end



### conventional Wolfe.

# modified Wolfe condition.
# equations 1.10, 1.11 from (Yuan 2017).
struct Wolfe{T}
    c1::T # this is δ in eqn 1.8.
    c2::T # this is σ in eqn 1.9.
end

function evalwolfeconditions(
    condition::Wolfe{T},
    ϕ_a::T,
    dϕ_a::T,
    a::T,
    u::Vector{T},
    ϕ_0::T, # f_x
    dϕ_0::T, # dot(df_x, u)
    )::Tuple{Bool,Bool} where T

    # parse.
    c1, c2 = condition.c1, condition.c2

    # checks.
    @assert zero(T) < c1 < c2 < one(T)

    # check these in the calling routine.
    #@assert dϕ_0 < zero(T) 
    #@assert isfinite(ϕ_0)
    #@assert isfinite(ϕ_a)

    # the condition for deciding whether the step size is too large.
    RHS1 = ϕ_0 + c1*a*dϕ_0 #+ a*min( -δ1*dϕ_0, c1*a*norm_u_sq/2)
    chk1 = ϕ_a <= RHS1

    # the condition for deciding whether the step size is too small.
    RHS2 = c2*dϕ_0 #+ min( -δ1*dϕ_0, c1*a*norm_u_sq)
    chk2 = dϕ_a >= RHS2

    return chk1, chk2
end