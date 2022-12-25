
abstract type βConfig end
# see cg_flavours.jl for the concrete types of `βConfig`.

abstract type TraceTrait end
struct EnableTrace <: TraceTrait end
struct DisableTrace <: TraceTrait end

#SystemsTrait(::Y) where Y <: MoleculeParams = Shared()
#SystemsTrait(::T) where T <: Nothing = DisableTrace()

# only stores a full, successful cg update.
struct TraceContainer{T,ET}
    objective::Vector{T}
    grad_norm::Vector{T}
    step_size::Vector{T}
    objective_evals::Vector{Int}
    status::ET
end

function setuptrace(::Type{T}, status::ET) where {T <: AbstractFloat, ET <: TraceTrait}
    return TraceContainer(
        Vector{T}(undef, 0),
        Vector{T}(undef, 0),
        Vector{T}(undef, 0),
        Vector{Int}(undef, 0),
        status,
    )
end

function resizetrace!(
    t::TraceContainer{T, EnableTrace},
    N::Int,
    ) where T
    
    resize!(t.objective, N)
    resize!(t.grad_norm, N)
    resize!(t.step_size, N)
    resize!(t.objective_evals, N)

    return nothing
end

function resizetrace!(
    ::TraceContainer{T, DisableTrace},
    args...
    ) where T

    return nothing
end

function updatetrace!(
    t::TraceContainer{T,EnableTrace},
    f_x::T,
    df_x_norm::T,
    step_size::T,
    objective_evals::Int,
    n::Int,
    ) where T

    t.objective[n] = f_x
    t.grad_norm[n] = df_x_norm
    t.step_size[n] = step_size
    t.objective_evals[n] = objective_evals

    return nothing
end

function updatetrace!(
    ::TraceContainer{T,DisableTrace},
    args...
    ) where T

    return nothing
end


### generic linesearch intermediate quantities.

struct LineSearchContainer{T} # all arrays have same size.
    xp::Vector{T}  # proposed iterate.
    df_xp::Vector{T}
    
    x::Vector{T} # current iterate.
    u::Vector{T} # search direction.
end

function LineSearchContainer(::Type{T}, N::Int) where T <: AbstractFloat

    return LineSearchContainer(
        Vector{T}(undef, N),
        Vector{T}(undef, N),
        Vector{T}(undef, N),
        Vector{T}(undef, N),
    )
end


##################

# if initial iterate meet the stopping criteria, treat as ran 0 iterations; no trace.
# if error, returns the most recent successful iterate.
mutable struct Results{T,TrT}
    objective::T
    minimizer::Vector{T}
    gradient::Vector{T}
    iters_ran::Int # if error, returns the number of successful iterations
    status::Symbol
    trace::TrT
end

function updateresult!(
    ret::Results{T,TrT},
    info::LineSearchContainer{T},
    f_x::T,
    i::Int,
    status::Symbol) where {T,TrT}

    ret.objective = f_x
    ret.minimizer[:] = info.xp
    ret.gradient[:] = info.df_xp
    ret.iters_ran = i
    ret.status = status

    resizetrace!(ret.trace, i)

    return nothing
end

function updateresult!(
    ret::Results{T,TrT},
    x::Vector{T},
    df_x::Vector{T},
    f_x::T,
    i::Int,
    status::Symbol) where {T,TrT}

    ret.objective = f_x
    ret.minimizer[:] = x
    ret.gradient[:] = df_x
    ret.iters_ran = i
    ret.status = status

    resizetrace!(ret.trace, i)

    return nothing
end

############# algorithm configurations.


struct CGConfig{T,BT,ET}
    # ρ::T # 0 < ρ < 1
    # σ::T # σ > 0
    # s::T # s > 0 
    ϵ::T # 0 < ϵ < 1
    β_config::BT
    #μ::T # 0 < μ < 1
    # linesearch_max_iters::Int
    max_iters::Int
    verbose::Bool
    trace_status::ET
    #linesearch_config::LT
end

# use default values from paper.
function setupCGConfig(
    ϵ::T,
    β_config::BT,
    trace_status::ET;
    #ρ = convert(T, 0.95),
    #σ = convert(T, 0.5),
    #s = convert(T, 1.0),
    #μ = convert(T, 0.1),
    #linesearch_max_iters = round(Int, log(ρ, 1e-6)),
    max_iters = 1000,
    verbose = false,
    )::CGConfig{T,BT,ET} where {T <: AbstractFloat, BT <: βConfig, ET <: TraceTrait}

    # @assert zero(T) < ρ < one(T)
    # @assert ρ > zero(T)
    # @assert s > zero(T)
    @assert zero(T) < ϵ < one(T)
    #@assert zero(T) < μ < one(T)

    return CGConfig(
        # ρ,
        # σ,
        # s,
        ϵ,
        β_config,
        #μ,
        #linesearch_max_iters,
        max_iters,
        verbose,
        trace_status,
        #linesearch_config,
    )
end


