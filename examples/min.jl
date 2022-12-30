

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



c1 = 1e-5
c2 = 0.8
linesearch_config = ConjugateGradientOptim.setupStrongWolfeBisection(
    c1, c2; a_max_growth_factor = 2.0,
    max_iters = 1000, zoom_max_iters = 100,
)

μ = 0.1
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


x0 = [0.43; 1.23]
ret = ConjugateGradientOptim.minimizeobjective(
    fdf!, x0, config, linesearch_config,
)

println("Results:")
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
