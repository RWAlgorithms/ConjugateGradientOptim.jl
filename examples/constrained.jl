

# using LinearAlgebra
# import Random
# using BenchmarkTools
# import LightSPSA

#Random.seed!(25)
#Random.seed!(35)
Random.seed!(5)

include("./helpers/test_funcs.jl")
#include("./helpers/utils.jl")

#H = LightSPSA.generateHadamardmatrix(8)

D = 6

max_iters = 20000
N_batches = 2000


Random.seed!(24)

PyPlot.close("all")
fig_num = 1


##########

fdf! = boothfdf!

function boxhdh!(
    fi_evals::Vector{T}, #mutates. length m. constraints.
    dfi_evals::Vector{Vector{T}}, #mutates, length m. gradient of constraints.
    x::Vector{T},
    lbs::Vector{T},
    ubs::Vector{T},
    ) where T
    
    @assert length(lbs) == length(ubs) # for box, number of constraints is 2*d.

    #
    fill!(fi_evals, zero(T))
    fi_evals[begin:begin+length(x)-1] = x .- ubs
    fi_evals[begin+length(x):end] = lbs .- x

    # the upper bounds.
    for d = 1:length(x)
        i = d
        fill!(dfi_evals[i], zero(T))
        dfi_evals[i][d] = one(T)
    end

    # the lower bounds.
    for d = 1:length(x)
        i = d + length(x)
        fill!(dfi_evals[i], zero(T))
        dfi_evals[i][d] = -one(T)
    end

    return nothing
end

hdh! = (fi_xx,dfi_xx,xx)->boxhdh!(
    fi_xx,
    dfi_xx,
    xx,
    lbs,
    ubs,
)

lbs = ones(2) .* -10
ubs = ones(2) .* 10

c1 = 1e-5
c2 = 0.8
linesearch_config = ConjugateGradientOptim.setupLinesearchNocedal(
    c1, c2; a_max_growth_factor = 2.0,
    max_iters = 1000, zoom_max_iters = 50,
)


ϵ = 1e-5
config = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    ConjugateGradientOptim.SallehAlhawarat(),
    #ConjugateGradientOptim.LiuStorrey(),
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
    verbose = false,
)

rerun_config1 = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    ConjugateGradientOptim.HagerZhang(),
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
    verbose = false,
)

μ = 0.1
rerun_config2 = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    ConjugateGradientOptim.setupYuanWangSheng(μ),
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
    verbose = false,
)

N_vars = 2
constraints = ConjugateGradientOptim.setupCvxInequalityConstraint(Float64, 2*N_vars, 2) # box constraint.

# TODO add verbose mode as dispatch?

x0 = [0.43; 1.23]
df_x0 = similar(x0)
f_x0 = fdf!(df_x0, x0)

u = hdh!(constraints.fi_evals, constraints.dfi_evals, x0)

# ## debug.
# @show constraints.fi_evals
# @show constraints.dfi_evals
# @show constraints.grad
# println()
# @show x0

# # I am here. track barriermethod() because it isn't matching up.
# #t0 = 1.0
# t0 = NaN
# t0 = ConjugateGradientOptim.verifyt0(
#     t0, 
#     x0, fdf!, 10.0, 0.0)
    
# #
# T = Float64
# f0df0! = fdf!

# t = ones(T, 1)
# t[begin] = t0
# qdq! = (gg,xx)->ConjugateGradientOptim.evalbarrier!(
#     constraints,
#     gg,
#     f0df0!,
#     hdh!,
#     xx,
#     t[begin],
# )


# dq_x0 = similar(x0)

# @show x0
# q_x0 = qdq!(dq_x0, x0)
# @show x0 # I am here. problem.

# @show constraints.fi_evals
# @show constraints.dfi_evals
# @show constraints.grad
# println()

# rets = ConjugateGradientOptim.minimizeobjectivererun(
#     qdq!,
#     x0,
#     config,
#     linesearch_config,
#     (rerun_config1, linesearch_config),
#     (rerun_config2, linesearch_config),
# )
# @show rets[end].status, rets[end].minimizer, length(rets)

# @assert 1==2

# ret = ConjugateGradientOptim.minimizeobjective(
#     #fdf!,
#     qdq!,
#     x0,
#     config,
#     linesearch_config,
# )
# @show ret.status, ret.minimizer

# ret2 = ConjugateGradientOptim.minimizeobjective(
#     #fdf!,
#     qdq!,
#     ret.minimizer,
#     config,
#     linesearch_config,
# )
# dq_x_star = similar(ret2.minimizer)
# q_x_star = qdq!(dq_x_star, ret2.minimizer)

# @show ret2.status, ret2.minimizer

# @assert 1==2

barrier_tol = 1e-8
barrier_growth_factor = 10.0
max_iters = 100
barrier_config = ConjugateGradientOptim.setupBarrierConfig(
    barrier_tol,
    barrier_growth_factor,
    max_iters,
)
b_ret = ConjugateGradientOptim.barriermethod!(
    constraints,
    fdf!,
    hdh!,
    x0,
    config,
    linesearch_config,
    barrier_config,
    (rerun_config1, linesearch_config),
    (rerun_config2, linesearch_config),
)

@show b_ret.status
@show b_ret.centering_results[end][end].status

@assert 6==5

rets = b_ret.centering_results
ret = rets[end]
@show b_ret.status, b_ret.iters_ran
println()

println("Last centering step's results:")
@show ret.minimizer, ret.objective, norm(ret.gradient), ret.status
@show sum(ret.trace.objective_evals), ret.iters_ran
println()

f_x_trace = ret.trace.objective
df_x_norm_trace = ret.trace.grad_norm
objective_evals = ret.trace.objective_evals

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(1:length(f_x_trace), f_x_trace,)

PyPlot.legend()
PyPlot.xlabel("iter")
PyPlot.ylabel("objective")
PyPlot.title("objective vs. iter")

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(1:length(objective_evals), objective_evals,)

PyPlot.legend()
PyPlot.xlabel("iter")
PyPlot.ylabel("linesearch_iters")
PyPlot.title("linesearch_iters vs. iter")




x_oracle = [1.0; 3.0]
df_x = ones(length(x_oracle))
f_x_oracle = fdf!(df_x, x_oracle)
println("Gradient at oracle:")
@show df_x
println()



# gradient check.
Random.seed!(343242)
x_test = randn(2)
f_x = fdf!(df_x, x_test)
@show f_x, df_x

ND_accuracy_order = 8

objectivefunc = xx->fdf!(df_x, xx)
df_ND = pp->FiniteDifferences.grad(
    FiniteDifferences.central_fdm(
        ND_accuracy_order,
        1,
    ),
    objectivefunc,
    pp
)[1]

df_ND_eval = df_ND(x_test)
@show norm(df_ND_eval)
println()

f_x = fdf!(df_x, x_test)
@show norm(df_x-df_ND_eval) # passed.

# next, implement options for PRP, HS, or LS β.
# next, implement bisection search from {s, ..., s*ρ^(i-1)}
# https://en.wikipedia.org/wiki/Bisection_method#Iteration_tasks
