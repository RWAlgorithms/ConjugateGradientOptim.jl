
struct BarrierResults{T,TrT}
    centering_results::Vector{Vector{Results{T,TrT}}}
    status::Symbol
    iters_ran::Int
    t_final::T
end

function assembleresults!(
    rets::Vector{Vector{Results{T,TrT}}},
    status::Symbol,
    iter::Int,
    t::T,
    )::BarrierResults{T,TrT} where {T,TrT}

    resize!(rets, iter)

    return BarrierResults(
        rets,
        status,
        iter,
        t
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

"""
```
setupCvxInequalityConstraint(::Type{T}, M::Int, D::Int)::CvxInequalityConstraint{T} where T
```
For strict inequality constraints.
`M` is the number of constraints.
`D` is the number of variables.
"""
function setupCvxInequalityConstraint(::Type{T}, M::Int, D::Int)::CvxInequalityConstraint{T} where T
    
    return CvxInequalityConstraint(
        Vector{T}(undef, M),
        collect( Vector{T}(undef, D) for _ = 1:M ),
        Vector{T}(undef, D),
    )
end

function getNconstraints(X::CvxInequalityConstraint{T})::Int where T
    return length(X.fi_evals)
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
    clamp!(fi_evals, convert(T, -Inf), zero(T))
    ψ_x = -sum( log(-fi_evals[i]) for i in eachindex(fi_evals)) # pg 563.

    fill!(dψ_x, zero(T))
    for i in eachindex(dfi_evals)
        for d in eachindex(dψ_x)
            dψ_x[d] -= dfi_evals[i][d]/fi_evals[i] # pg 564.
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
    #assignbarriergrad!(df_x, constraints.grad, t)
    df_x[:] = t .* df_x .+ constraints.grad

    return t*f_x + ψ_x # return the barrier objective.
end

struct BarrierConfig{T}
    barrier_tol::T
    barrier_growth_factor::T
    max_iters::Int
    t_initial::T
    inf_f0_lb::T
end

function setupBarrierConfig(
    barrier_tol::T,
    barrier_growth_factor::T,
    max_iters::Int,
    ) where T

    t_initial = convert(T, NaN)
    inf_f0_lb = zero(T)

    return BarrierConfig(
        barrier_tol,
        barrier_growth_factor,
        max_iters,
        convert(T, NaN),
        inf_f0_lb,
    )
end

# algorithm 11.1 (Boyd 2004).
function barriermethod!(
    constraints::CvxInequalityConstraint{T},
    f0df0!,
    hdh!,
    x_initial::Vector{T},
    centering_config::CGConfig{T,BT,ET},
    linesearch_config::LinesearchNocedal{T},
    barrier_config::BarrierConfig{T},
    rerun_config_tuples...
    ) where {T,BT,ET}
    
    # parse.
    barrier_tol = barrier_config.barrier_tol
    barrier_growth_factor = barrier_config.barrier_growth_factor
    max_iters = barrier_config.max_iters
    t_initial = barrier_config.t_initial
    inf_f0_lb = barrier_config.inf_f0_lb

    N_constraints = getNconstraints(constraints)

    # set up buffers.
    x = copy(x_initial)
    # D = length(x_initial)
    #df_x = Vector{T}(undef, D)

    # outputs.
    rets = Vector{Vector{Results{T,TraceContainer{T,ET}}}}(undef, max_iters)

    # check if the initial iterate is feasible.
    hdh!(constraints.fi_evals, constraints.dfi_evals, x)

    if any(constraints.fi_evals .>= zero(T))
        println("Infeasible start. Exit.")

        return assembleresults!(
            rets,
            :infeasible_start,
            0,
            t_initial,
        )
    end

    t_initial = verifyt0(t_initial, x_initial, f0df0!, barrier_growth_factor, inf_f0_lb)
    #@show t_initial

    # set up objective function for each centering step.
    t = ones(T, 1)
    t[begin] = t_initial
    fdf! = (gg,xx)->evalbarrier!(
        constraints,
        gg,
        f0df0!,
        hdh!,
        xx,
        t[begin],
    )

    for i = 1:max_iters

        rets[i] = minimizeobjectivererun(
            fdf!,
            x,
            centering_config,
            linesearch_config,
            rerun_config_tuples...
        )
        if rets[i][end].status != :success

            return assembleresults!(
                rets,
                :centering_step_issue,
                i,
                t[begin],
            )
        end

        # stopping condition.
        if N_constraints/t[begin] < barrier_tol

            return assembleresults!(
                rets,
                :success,
                i,
                t[begin],
            )
        end

        # book keep.
        t[begin] = barrier_growth_factor*t[begin]
    end

    return assembleresults!(
        rets,
        :max_iters_reached,
        max_iters,
        t[begin],
    )
end

# TODO: better heurestic to estimate the initial t value for the barrier method.
# select initial t0 via page 570-571 of (Boyd 2004).
function verifyt0(
    t0::T,
    x0::Vector{T},
    f0df0!,
    μ::T,
    inf_f0_lb::T,
    )::T where T

    if !isfinite(t0) || t0 < zero(T)
        
        df_x0 = Vector{T}(undef, length(x0))
        f_x0 = f0df0!(df_x0, x0)
        t0 = (f_x0 - inf_f0_lb) * μ

        return t0
    end

    return t0
end

# note the use of a barrier function NEEDS all line search functions or 
#   any function evals to never violate the constraint.