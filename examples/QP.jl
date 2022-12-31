Random.seed!(5)

include("./helpers/test_funcs.jl")

PyPlot.close("all")
fig_num = 1

##########
G, c, A, b, x_oracle, λ_oracle = equalityQPexample169()
problem = ConjugateGradientOptim.EqualityQP(G,c,A,b)
x_star, status = ConjugateGradientOptim.solveequalityQP(problem)

@show x_star, status
@show norm(x_star - x_oracle)