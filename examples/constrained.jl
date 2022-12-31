

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
N_vars = 2

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

lbs = ones(N_vars) .* -10
ubs = ones(N_vars) .* 10

# c1 = 1e-5
# c2 = 0.8
# linesearch_config = ConjugateGradientOptim.setupStrongWolfeBisection(
#     c1, c2; a_max_growth_factor = 2.0,
#     max_iters = 1000, zoom_max_iters = 50,
# )

c1 = 1e-3
c2 = 0.9
δ1 = 1e-6
condition = ConjugateGradientOptim.YuanWeiLuWolfe(c1,c2,δ1)
#condition = ConjugateGradientOptim.Wolfe(c1,c2)
max_iters = 50
max_step_size = 1e12
feasibility_max_iters = 50
linesearch_config = ConjugateGradientOptim.WolfeBisection(
    condition, max_iters, max_step_size, feasibility_max_iters,
)

max_iters2 = 300
discount_factor = 0.9
condition2 = ConjugateGradientOptim.Armijo(c1)
linesearch_config2 = ConjugateGradientOptim.Backtracking(
    condition2, discount_factor, max_iters2, feasibility_max_iters,
)

μ = 0.1
#ϵ_coarse = 1e-3
ϵ = 1e-5
config = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    #ConjugateGradientOptim.setupYuanWangSheng(μ),
    ConjugateGradientOptim.HagerZhang(),
    #ConjugateGradientOptim.SallehAlhawarat(),
    #ConjugateGradientOptim.LiuStorrey(),
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
    verbose = false,
)

#B_config = ConjugateGradientOptim.setupBroydenFamily(0.0, N_vars) # BFGS
B_config = ConjugateGradientOptim.setupBroydenFamily(1.0, N_vars) # DFP
rerun_config1 = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    B_config,
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
    verbose = false,
)

rerun_config2 = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    #ConjugateGradientOptim.setupYuanWangSheng(μ),
    ConjugateGradientOptim.LiuStorrey(),
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
    verbose = false,
)


constraints = ConjugateGradientOptim.setupCvxInequalityConstraint(Float64, 2*N_vars, 2) # box constraint.

# TODO add verbose mode as dispatch?

x0 = [0.43; 1.23]
df_x0 = similar(x0)
f_x0 = fdf!(df_x0, x0)

u = hdh!(constraints.fi_evals, constraints.dfi_evals, x0)


barrier_tol = 1e-8
barrier_growth_factor = 10.0
max_iters = 100
barrier_config = ConjugateGradientOptim.setupPrimalBarrierConfig(
    barrier_tol,
    barrier_growth_factor,
    max_iters;
    #t_initial = 1.0,
)
b_ret = ConjugateGradientOptim.primalbarriermethod!(
    constraints,
    fdf!,
    hdh!,
    x0,
    config,
    linesearch_config,
    barrier_config,
    (rerun_config1, linesearch_config2), # enabling quasi-Newton actually makes it slower to converge, if the Wolfe condition for linesearch is used. better with Armijo condition.
    (rerun_config2, linesearch_config),
)

@show b_ret.status
@show b_ret.total_objective_evals
@show b_ret.centering_results[end][end].status
@show b_ret.centering_results[end][end].minimizer
@show norm(b_ret.centering_results[end][end].gradient)
rets = b_ret.centering_results

obj_evals = collect(collect(rets[i][j].trace.objective_evals for j in eachindex(rets[i])) for i in eachindex(rets))

ret = rets[end][end]
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
println("zero if non-binding constraints.")
println()

