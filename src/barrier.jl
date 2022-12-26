
struct BarrierResults{T,TrT}
    centering_results::Vector{Results{T,TrT}}
    status::Symbol
    iters_ran::Int
end

function assembleresults!(
    rets::Vector{Results{T,TrT}},
    status::Symbol,
    iter::Int,
    )::BarrierResults{T,TrT} where {T,TrT}

    resize!(rets, iter)

    return BarrierResults(
        rets,
        status,
        iter,
    )
end

# buffers intended for the barrier method.
# struct BoxConstraint{T}
#     fi_evals::Vector{T}
#     dfi_evals::Vector{Vector{T}}
#     grad::Vector{T}
# end

# struct OrderConstraint{T}
#     fi_evals::Vector{T}
#     dfi_evals::Vector{Vector{T}}
#     grad::Vector{T}
# end

struct CvxInequalityConstraint{T}
    fi_evals::Vector{T}
    dfi_evals::Vector{Vector{T}}
    grad::Vector{T}
end

# generic barrier algorithm.
# the general case. sec 11.2.1 (Boyd 2004).
# ψ here is ϕ in (Boyd 2004).
# in standard form, i.e. h[i](x) <= 0 is one of the inequality constraints.
function evalconstraints!(
    constraints::CvxInequalityConstraint{T},
    hdh!,
    x::Vector{T},
    ) where T

    fi_evals, dfi_evals = constraints.fi_evals, constraints.dfi_evals
    dψ_x = constraints.grad
    @assert length(fi_evals) == length(dfi_evals)

    hdh!(fi_evals, dfi_evals, x)

    # For real arguments, Julia's log() only take non-negative values. Otherwise it throws a domain error. Need to manually clamp negative values to 0, and rely on Julia's definition: log(0) = -Inf.
    clamp!(fi_evals, zero(T), convert(T,Inf))
    ψ_x = -sum( log(-fi_evals[i]) for i in eachindex(fi_evals)) # pg 563.

    fill!(dψ_x, zero(T))
    for i in eachindex(dfi_evals)
        for d in eachindex(dψ_x)
            dψ_x[d] -= dfi_evals[i]/fi_evals[i] # pg 564.
        end
    end
    
    return ψ_x
end

function assignbarriergrad!(
    df_x::Vector{T},
    dψ_x::Vector{T},
    t::T
    ) where T

    @assert length(df_x) == length(dψ_x)

    for d in eachindex(df_x)
        df_x[d] = t*df_x[d] + dψ_x[d]
    end

    return nothing
end

function evalbarrier!(
    constraints::CvxInequalityConstraint{T},
    df_x::Vector{T},
    fdf!,
    hdh!,
    x::Vector{T},
    t::T,
    ) where T

    # evaluate f0 and constraints.
    f_x = fdf!(df_x, x)
    ψ_x = evalconstraints!(constraints, hdh!, x)

    # their contribution to the barrier gradient.
    assignbarriergrad!(df_x, constraints.grad, t)

    return t*f_x + ψ_x # return the barrier objective.
end

# algorithm 11.1 (Boyd 2004).
function barriermethod!(
    constraints::CvxInequalityConstraint{T},
    f0df0!,
    hdh!,
    x_initial::Vector{T},
    t_initial::T,
    centering_config::CGConfig{T,ET},
    linesearch_config::LinesearchNocedal{T};
    barrier_tol::T = convert(T, 1e-8),
    barrier_growth_factor::T = convert(T, 10),
    max_iters::Int = convert(Int, 300),
    ) where {T,ET}
    
    # parse.
    N_constraints = length(constraints.fi_evals)
    D = length(x_initial)

    # set up buffers.
    x = copy(x_initial)
    df_x = Vector{T}(undef, D)

    # outputs.
    rets = Vector{Results{T,TraceContainer{T,ET}}}(undef, max_iters)

    # check if the initial iterate is feasible.
    h_x = hdh!(constraints.grad, x)
    if h_x >= zero(T)
        println("Infeasible start. Exit.")

        return assembleresults!(
            rets,
            :infeasible_start,
            0,
        )
    end

    # set up objective function for each centering step.
    t = ones(T, 1)
    t[begin] = t_initial
    fdf! = xx->evalbarrier!(
        constraints,
        df_x,
        f0df0!,
        hdh!,
        xx,
        t[begin],
    )

    for i = 1:max_iters

        rets[i] = minimizeobjective(
            fdf!,
            x,
            centering_config,
            linesearch_config,
        )
        if rets[i].status != :success

            return assembleresults!(
                rets,
                :centering_step_issue,
                i,
            )
        end

        # stopping condition.
        if N_constraints/t < barrier_tol

            return assembleresults!(
                rets,
                :success,
                i,
            )
        end

        # book keep.
        t[begin] = barrier_growth_factor*t[begin]
    end

    return assembleresults!(
        rets,
        :max_iters_reached,
        max_iters,
    )
end




# note the use of a barrier function NEEDS all line search functions or 
#   any function evals to never violate the constraint.