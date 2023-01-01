# ConjugateGradientOptim.jl
Conjugate gradient and quasi-Newton solvers for differentiable unconstrained optimization. Lightweight dependencies, stable interface, real-valued objection functions.

Failures are handled gracefully.

Examples are in `./examples/`.

Auto-restart with different algorithms are available via `minimizeobjectivererun`.

A primal barrier method is implemented for inequality constrained optimization, but it is not recommended to use this in actual application. This is because the primal barrier method successively solves harder and harder unconstrained optimization problems, to the point that linesearch failures are common due to finite numerical precision.