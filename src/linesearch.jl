# equations 1.10, 1.11 from (Yuan 2017).
struct LinesearchYuanWeiLu{T}
    c1::T # this is δ in eqn 1.8.
    c2::T # this is σ in eqn 1.9.
    δ1::T # this is δ1 in eqn 1.10, 1.11.
end

struct GeometricLinesearch{CT}
    condition::CT
    decrease_factor::T
    increase_factor::T
    max_iters::Int
end

function linesearch!(
    info::LineSearchContainer{T},
    config::GeometricLinesearch{T},
    fdf!,
    f_x::T,
    df_x::Vector{T},
    a_initial::T;
    verbose = false,
    ) where T <: AbstractFloat

    # parse.
    xp, df_xp, x, u = info.xp, info.df_xp, info.x, info.u

end

function evallinesearchcondition!(
    df_xp::Vector{T},
    xp::Vector{T},
    condition::LinesearchYuanWeiLu{T},
    fdf!,
    fdf_evals_ran::Int,
    a::T,
    x::Vector{T},
    u::Vector{T},
    ϕ_0::T, # f_x
    dϕ_0::T, # dot(df_x, u)
    )::Tuple{Bool,Int} where T

    # parse.
    c1, c2, δ1 = condition.c1, condition.c2, condition.δ1
    @assert zero(T) < δ1 < c1

    # evaluate.
    ϕ_a, dϕ_a = evalϕdϕ!(xp, df_xp, fdf!, a, x, u)
    fdf_evals_ran += 1

    norm_u_sq = dot(u,u)

    RHS1 = ϕ_0 + c1*a*dϕ_0 + a*min( -δ1*dϕ_0, c1*a*norm_u_sq/2)
    chk1 = ϕ_a <= RHS1

    RHS2 = c2*dϕ_0 + min( -δ1*dϕ_0, c1*a*norm_u_sq)
    chk2 = dϕ_a >= RHS2

    return chk1 && chk2, fdf_evals_ran
end

# I am here. compare Algorithm 0 with linesearch!()
# to see if we can swap out the linesearch condition.


######### WIP

struct BisectionLinesearch{CT}
    condition::CT
    max_iters::Int
end

# TODO: work in progress.
function linesearch!(
    info::LineSearchContainer{T},
    config::BisectionLinesearch{T},
    fdf!,
    f_x::T,
    df_x::Vector{T},
    a_initial::T;
    verbose = false,
    ) where T <: AbstractFloat

    # parse.
    xp, df_xp, x, u = info.xp, info.df_xp, info.x, info.u

end