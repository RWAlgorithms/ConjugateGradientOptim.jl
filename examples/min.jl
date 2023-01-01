
include("a.jl") # disable if running this script repeately.

# set up Julia session.
include("./helpers/test_funcs.jl")

Random.seed!(24)
PyPlot.close("all")
fig_num = 1


# objective: see /examples/helpers/test_funcs.jl.
fdf! = boothfdf!

# line search configuration.
c1 = 1e-5
c2 = 0.8
linesearch_config = ConjugateGradientOptim.setupStrongWolfeBisection(
    c1, c2; a_max_growth_factor = 2.0,
    max_iters = 1000, zoom_max_iters = 100,
)

# conjugate gradient configuration.
μ = 0.1
ϵ = 1e-5
config = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    #ConjugateGradientOptim.YuanWangSheng(μ),
    ConjugateGradientOptim.HagerZhang(),
    #ConjugateGradientOptim.SallehAlhawarat(),
    #ConjugateGradientOptim.LiuStorrey(),
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
    verbose = false,
)

# initial iterate.
x0 = [0.43; 1.23]

# run solver.
ret = ConjugateGradientOptim.minimizeobjective(
    fdf!, x0, config, linesearch_config,
)

# visualize results.
println("Results:")
@show ret.minimizer, ret.objective, norm(ret.gradient), ret.status
@show sum(ret.trace.objective_evals), ret.iters_ran
println()

f_x_trace = ret.trace.objective
df_x_norm_trace = ret.trace.grad_norm
objective_evals = ret.trace.objective_evals

# visualize solver run diagnostics.
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