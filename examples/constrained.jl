
include("a.jl") # disable if running this script repeately.

# Set up Julia session.
include("./helpers/test_funcs.jl")

Random.seed!(24)
PyPlot.close("all")
fig_num = 1

# # Set up
N_vars = 2

# `fdf!` is the objective function.
fdf! = boothfdf!

# ## Box constraint
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

# constraint function.

lbs = ones(N_vars) .* -10 # lower bounds.
ubs = ones(N_vars) .* 10 # upper bounds.

# assemble the constraint function, `hdh!`.
hdh! = (fi_xx,dfi_xx,xx)->boxhdh!(
    fi_xx,
    dfi_xx,
    xx,
    lbs,
    ubs,
)

# # Configurations

# ## Line search configurations.
max_iters_ls = 100
max_step_size = 1e12
feasibility_max_iters = 50

# Strong Wolfe conditions. Uses a variation of Algorithm 3.6 from Nocedal 2006.
c1 = 1e-5
c2 = 0.8
linesearch_config_SW = ConjugateGradientOptim.setupStrongWolfeBisection(
    c1, c2; a_max_growth_factor = 2.0,
    max_iters = max_iters_ls, zoom_max_iters = 50,
)

# (weak) Wolfe conditions. Uses a bisection algorithm.
c1 = 1e-3
c2 = 0.9
condition = ConjugateGradientOptim.Wolfe(c1,c2)
linesearch_config_W = ConjugateGradientOptim.WolfeBisection(
    condition, max_iters_ls, max_step_size, feasibility_max_iters,
)


# Yuan's modification of the (weak) Wolfe conditions.
δ1 = 1e-6
condition = ConjugateGradientOptim.YuanWeiLuWolfe(c1,c2,δ1)
max_iters = 50
max_step_size = 1e12
feasibility_max_iters = 50
linesearch_config_YW = ConjugateGradientOptim.WolfeBisection(
    condition, max_iters_ls, max_step_size, feasibility_max_iters,
)

# Armijo condition.
max_iters2 = 300
discount_factor = 0.9
condition2 = ConjugateGradientOptim.Armijo(c1)
linesearch_config_A = ConjugateGradientOptim.Backtracking(
    condition2, discount_factor, max_iters2, feasibility_max_iters,
)

# ## Conjugate gradient configurations.

ϵ = 1e-5 # stopping condition: gradient norm residual.

config_HZ = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    ConjugateGradientOptim.HagerZhang(),
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
)

μ = 0.1
config_YWS = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    ConjugateGradientOptim.YuanWangSheng(μ),
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
)

config_LS = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    ConjugateGradientOptim.LiuStorrey(),
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
)

config_SA = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    ConjugateGradientOptim.SallehAlhawarat(),
    #ConjugateGradientOptim.LiuStorrey(),
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
)

Broyden_DFP = ConjugateGradientOptim.setupBroydenFamily(1.0, N_vars) # DFP
#Broyden_BFGS = ConjugateGradientOptim.setupBroydenFamily(0.0, N_vars) # BFGS
config_Broyden_DFP = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    Broyden_DFP,
    ConjugateGradientOptim.EnableTrace();
    max_iters = 1000,
)

# # Run solver.

# Select line search and CG configurations.
linesearch_config = linesearch_config_W
linesearch_config1 = linesearch_config_YW
linesearch_config2 = linesearch_config_A

config = config_HZ
rerun_config1 = config_Broyden_DFP
rerun_config2 = config_LS
# enabling quasi-Newton actually makes it slower to converge for this primal barrier problem, if the Wolfe condition for linesearch is used. better with Armijo condition.

# Initial iterate.
x0 = [0.43; 1.23]
println("Initial iterate: ", x0)
println("Global minimum: ", [1; 3])
println()

# (optional) Get the initial objective and gradient.
df_x0 = similar(x0)
f_x0 = fdf!(df_x0, x0)

# Allocate buffer for constraints.
constraints = ConjugateGradientOptim.setupCvxInequalityConstraint(Float64, 2*N_vars, 2) # box constraint.

# (optional) evaluate the constraints and gradient at initial iterate.
h_x0 = hdh!(constraints.fi_evals, constraints.dfi_evals, x0)

# Barrier configurations.
barrier_tol = 1e-8
barrier_growth_factor = 10.0
max_iters = 100
barrier_config = ConjugateGradientOptim.setupPrimalBarrierConfig(
    barrier_tol,
    barrier_growth_factor,
    max_iters;
    #t_initial = 1.0,
)

# run solver.
b_ret = ConjugateGradientOptim.primalbarriermethod!(
    constraints,
    fdf!,
    hdh!,
    x0,
    config,
    linesearch_config,
    barrier_config,
    (rerun_config1, linesearch_config2), 
    (rerun_config2, linesearch_config),
)

# # Visualize results.
@show b_ret.status
@show b_ret.total_objective_evals
@show b_ret.centering_results[end][end].status
@show b_ret.centering_results[end][end].minimizer
@show norm(b_ret.centering_results[end][end].gradient)
rets = b_ret.centering_results

obj_evals = collect(collect(rets[i][j].trace.objective_evals for j in eachindex(rets[i])) for i in eachindex(rets))

# Visualize the very last run of the algorithm, in the series of unconstrained problems that the barrier method solved.
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
