module ConjugateGradientOptim

using LinearAlgebra

#include("./linesearch/linesearch_types.jl")
#include("./linesearch/alg851.jl")

# TODO make sure every variable is passed properly, within scope.
# tests against every implemented cg flavour. Use setup from (Salleh 2016).
# for barriermethod(), write routine that allows retry with different method if stuck.
# TODO: verbose flag is not functional for now.

include("types.jl")
include("cg_utils.jl")
include("cg_flavours.jl")
include("qn_flavours.jl")

include("./linesearch/wolfe.jl")
include("./linesearch/nocedal.jl")

include("./engine/solve_system.jl")
include("./engine/optim.jl")
include("./engine/barrier.jl")

export TraceContainer,
EnableTrace,
DisableTrace,
Results,
LineSearchContainer,
solvesystem,
minimizeobjective

end # end of module

# # References

# ## (Salleh 2016)
# Salleh, Z., & Alhawarat, A. (2016). An efficient modification of the Hestenes-Stiefel nonlinear conjugate gradient method with restart property. Journal of Inequalities and Applications, 2016(1), 1-14.
# modified Hestenses-Stiefel so that it restarts if ascent search direction is encountered.
# DOI: 10.1186/s13660-016-1049-5

# ## (Yuan 2019)
# Yuan, G., Wang, B., & Sheng, Z. (2019). The Hager–Zhang conjugate gradient algorithm for large-scale nonlinear equations. International Journal of Computer Mathematics, 96(8), 1533-1547.
# modified Hager-Zhang to have trust region-like behavior.
# DOI: 10.1080/00207160.2018.1494825

# ## (Yuan 2021)
# Yuan, G., Lu, J., & Wang, Z. (2021). The modified PRP conjugate gradient algorithm under a non-descent line search and its application in the Muskingum model and image restoration problems. Soft Computing, 25(8), 5867-5879.
# modified Polak–Ribière–Polyak to deal with ascent search direction.
# DOI: 10.1007/s00500-021-05580-0

# ## (Nocedal 2006)
# Nocedal, J., & Wright, S. J. (2006). Numerical optimization.
# reference textbook on numerical optimization.
# DOI: 10.1007/978-0-387-40065-5

# ## (Yuan 2017)
# Yuan, G., Wei, Z., & Lu, X. (2017). Global convergence of BFGS and PRP methods under a modified weak Wolfe–Powell line search. Applied Mathematical Modelling, 47, 811-825.
# a modified weak Wolfe-Powell line search, known as the YWL line search.
# DOI: 10.1016/j.apm.2017.02.008
