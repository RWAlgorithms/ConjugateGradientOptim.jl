module ConjugateGradientOptim

using LinearAlgebra

#include("./linesearch/linesearch_types.jl")
#include("./linesearch/alg851.jl")


include("types.jl")
include("cg_utils.jl")
include("cg_flavours.jl")

include("solve_system.jl")
include("optim.jl")


export TraceContainer,
EnableTrace,
DisableTrace,
Results,
LineSearchContainer,
solvesystem,
minimizeobjective

end # module MinimalistConjGradOptim
