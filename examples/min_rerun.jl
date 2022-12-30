

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
c2 = 0.9
linesearch_config = ConjugateGradientOptim.setupStrongWolfeBisection(
    c1, c2; a_max_growth_factor = 2.0,
    max_iters = 1000, zoom_max_iters = 50,
)


ϵ = 1e-5
config = ConjugateGradientOptim.setupCGConfig(
    ϵ,
    #ConjugateGradientOptim.HagerZhang(),
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

x0 = [0.43; 1.23]
df_x0 = similar(x0)
f_x0 = fdf!(df_x0, x0)

u = hdh!(constraints.fi_evals, constraints.dfi_evals, x0)

#t0 = 1.0 # manually specify t0.
t0 = NaN # use verifyt0() to specify t0.
t0 = ConjugateGradientOptim.verifyt0(
    t0, 
    x0, fdf!, 10.0, 0.0)
    
#
f0df0! = fdf!

t = ones(Float64, 1)
t[begin] = t0
qdq! = (gg,xx)->ConjugateGradientOptim.evalbarrier!(
    constraints,
    gg,
    f0df0!,
    hdh!,
    xx,
    t[begin],
)

rets = ConjugateGradientOptim.minimizeobjectivererun(
    qdq!,
    x0,
    config,
    linesearch_config,
    (rerun_config1, linesearch_config),
    (rerun_config2, linesearch_config),
)
@show rets[end].status, rets[end].minimizer, length(rets)