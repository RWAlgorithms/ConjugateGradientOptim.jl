
# 0 < c1 <c2 < 1.
struct WolfeType{T}
    c1::T
    c2::T
end

# Eqn 3.6 of Nocedal 2006.
# scale-invariant: cost function multiple or affine change of variables do not change the conditions.
function linesearch(
    config::WolfType{T},
    α,
    p::Vector{T},
    f_x::T,
    df_x::Vector{T},
    x::Vector{T},
    ) where T <: AbstractFloat

    u = x + α .* p # u(α).

    f(u) <= f_x + c1*α* dot(df_x, p) # decrease.

    # Wolfe.
    dot(df(u), p) >= c2*dot(df_x, p) # curvature.

    # Strong Wolfe (modified curvature). Exclude α that are far from being being a tationary point of f(u(α)).
    abs(dot(df(u), p)) <= c2*abs(dot(df_x, p))

end

# Nocedal recommends not using this for CG.
function backtracksearch(α0::T, p::Vector{T}, x::Vector{T}, f, r::T, max_iters::Int) where T <: AbstractFloat

    @assert α0 > 0
    @assert c1 > 0
    @assert 0 < r < 1
    
    α = α0
    for _ = 1:max_iters

        u = x .+ α .* p
        if f(u) < f_x + c1*α* dot(df_x, p)
            return α
        end

        α = r*α
    end

    return -one(T)
end

# pg 77, sec 3.5. Assumes p is a descent direction.
function linesearch()
    
    u = x .+ α .* p
    @assert dot(df(u), p) < 0 # LHS is the derivative of f(u). Checks if p is a descent direction.

    return 
end

# https://link.springer.com/article/10.1007/s12065-022-00783-2?error=cookies_not_supported&code=8c1735a7-e672-445f-a8b3-3eee13eab034