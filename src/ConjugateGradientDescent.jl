module ConjugateGradientDescent

using LinearAlgebra

#include("./linesearch/linesearch_types.jl")
#include("./linesearch/alg851.jl")

include("linesearch_types.jl")
include("types.jl")
include("cg_utils.jl")

include("solve_system.jl")


export TraceContainer,
EnableTrace,
DisableTrace,
Results,
LineSearchContainer,
runcg


end # module MinimalistConjGradOptim
