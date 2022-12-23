

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

#booth function
f = xx->((xx[1]+2*xx[2]-7)^2+(2*x[1]+x[2]-5)^2)

function myfdf!(g_p::Vector{T}, p::Vector{T}) where T <: AbstractFloat
    x, y = p

    f_p = (x+2*y-7)^2 + (2*x+y-5)^2

    g_p[begin] = 2*(x+2*y-7) + 2*(2*x+y-5)*2
    g_p[begin+1] = 2*(x+2*y-7)*2 + 2*(2*x+y-5)

    return f_p
end

# I am here. diverging!

##########

fdf! = myfdf!

config = ConjugateGradientDescent.Yuan2019(
    0.5, 0.5, 0.95, 0.5, 0.5, 1000, max_iters)


x0 = [0.43; 1.23]
ret, f_x_trace = ConjugateGradientDescent.runcgyuan2019(
    fdf!, x0, config)


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(1:length(f_x_trace), f_x_trace,)


PyPlot.legend()
PyPlot.xlabel("iter")
PyPlot.ylabel("cost")
PyPlot.title("cost vs. iter")


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

costfunc = xx->fdf!(df_x, xx)
df_ND = pp->FiniteDifferences.grad(
    FiniteDifferences.central_fdm(
        ND_accuracy_order,
        1,
    ),
    costfunc,
    pp
)[1]

df_ND_eval = df_ND(x_test)
@show norm(df_ND_eval)
println()

f_x = fdf!(df_x, x_test)
@show norm(df_x-df_ND_eval) # passed.