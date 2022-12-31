# based on (Nocedal 2006), unless specified otherwise.

# eqn 15.1.
struct GeneralConstrainedProblem
    f!
    h!
    g!
end

struct SecondOrderBuffer{T}
    df_x::Vector{T}
    d2f_x::Matrix{T}
    h_x::Vector{T}
    dh_x::Vector{Vector{T}}
    d2h_x::Vector{Matrix{T}}
    g_x::Vector{T}
    dg_x::Vector{Vector{T}}
    d2g_x::Vector{Matrix{T}}
end

struct BarrierSubproblem
    μ::T
    s::Vector{T} #positive.
    x::Vector{T}
end

# struct InteriorPoint{T}
#     A_eq::Matrix{T}
#     A_ineq::Matrix{T}
#     s::Vector{T} # slack variables for turning inequalities to equalities. non-negative.
#     y::Vector{T} # Lagrange multipliers for equality constraints.
#     z::Vector{T} # Lagrange multipliers for inequality constraints. non-negative.
#     σ::T
#     τ::T # eqn 19.9
# end

function evalLagrangian!(
    buf::SecondOrderBuffer,
    problem::GeneralConstrainedProblem{T},
    subproblem::BarrierSubproblem{T},
    x::Vector{T},
    ) where T

    f!, g!, h! = problem.f!, problem.g!, problem.h!
    df_x, d2f_x = buf.df_x, buf.d2f_x
    h_x, dh_x, d2h_x = buf.h_x, buf.dh_x, buf.d2h_x
    g_x, dg_x, d2g_x = buf.g_x, buf.dg_x, buf.d2g_x

    f_x = f!(df_x, d2f_x, x)
    h!(h_x, dh_x, d2h_x, x)
    g!(g_x, dg_x, d2g_x, x)

    #dot(y, )

end

