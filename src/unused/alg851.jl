
struct Alg851Config{T}
    δ::T
    σ::T
end

function approximatewolfe()

    @assert 0 < δ < 1/2
    @assert δ <= σ < 1

    (2*δ-1)*dϕ(0) >= dϕ(α) >= σ*dϕ(0)

end


######## update interval (bracket) for line search.
# pg 11/25 of CG_DESCENT paper.

# eqn 29 of Alg851.
function oppositeslopecondition(
    ϕ_a,
    dϕ_a,
    dϕ_b,
    ϕ_0::T,
    ϵ::T,
    )::Bool where T <: AbstractFloat
    
    chk1 = ϕ_a <= ϕ_0 + ϵ
    chk2 = dϕ_a < 0
    chk3 = dϕ_b >= 0

    return chk1 && chk2 && chk3
end

# error-handling
function returninterval(lb::T, ub::T)::Tuple{T,T,Bool} where T
    if !(lb < ub)
        return lb, ub, false
    end

    return lb, ub, true
end

# [a,b] is the current bracketing interval, c is an update point.
# This is 11/25, interval update rules, in Alg851.
function updatebracketinverval!(
    info::LineSearchContainer{T},
    fdf!,
    a::T,
    b::T,
    c::T,
    ϕ_0::T,
    #dϕ_0::T
    #ϕ_a::T,
    #dϕ_a::T,
    #ϕ_b::T,
    #dϕ_b::T,
    ϕ_c::T,
    dϕ_c::T,
    ϵ::T;
    θ = convert(T,0.5),
    max_iters::Integer = 1000,
    ) where T <: AbstractFloat
    
    # pre-compute.
    ϕ0_plus_ϵ = ϕ_0 + ϵ
    cvx_w1 = one(T) - θ
    cvx_w2 = θ

    # U0.
    if !(a < c < b)
        return returninterval(a, b)
    end

    if dϕ_c < 0
        if ϕ_c > ϕ0_plus_ϵ
            # U3.
            a_new = a
            b_new = b

            for _ = 1:max_iters
                
                d = cvx_w1*a_new + cvx_w2*b_new

                ϕ_d, dϕ_d = evalϕdϕ!(info, fdf!, d)
                if dϕ_d < 0
                
                    if ϕ_d > ϕ0_plus_ϵ
                        # U3c.
                        b_new = d
                    else
                        # U3b.
                        a_new = d
                    end

                else
                     # U3a.
                     b_new = d
                     return returninterval(a_new, b_new)
                end
            end

            return a_new, b_new, false # line search reached maximum iterations, consider as a failed search.
        else
            
            # U2.
            return returninterval(c, b)
        end
    end

    # U1.
    return returninterval(a, c)
end

# # disable to reduce complexity. from the paragraph under U steps on pg 11.
# function checkinterval(
#     a::T,
#     b::T,
#     a_new::T,
#     b_new::T,
#     ϕ_a_new::T,
#     ϕ_b_new::T,
#     dϕ_a_new::T,
#     dϕ_b_new::T,
#     ϕ_0_plus_ϵ::T,
#     ) where T

#     if !(a <= a_new < b_new <= b)
#         return false
#     end

#     # these should be satisfied.
#     chk1 = (dϕ_a_new < 0)
#     chk2 = (ϕ_a_new <= ϕ_0_plus_ϵ)

#     chk3 = (dϕ_b_new < 0)
#     chk4 = (ϕ_b_new > ϕ_0_plus_ϵ)
    
#     return chk1 && chk2 && chk3 && chk4
# end

################ secant update as candidate for a bracket end point.
# pg 11/25 of Alg851

function computesecant(a::T, b::T, dϕ_a::T, dϕ_b::T)::T where T
    return a*dϕ_b - b*dϕ_a / (dϕ_b - dϕ_a)
end



# titled secant in Alg851.
# assumes a, b, satisfies oppositeslopecondition().
function updateinterval(
    q::LineSearchIntermediates{T},
    info::LineSearchContainer{T},
    config::LineSearchConfig{T},
    a::T,
    b::T,
    #ϕ_a::T,
    dϕ_a::T,
    #ϕ_b::T,
    dϕ_b::T,
    ϕ_0::T,
    ϵ::T,
    )::T where T

    # ϕ_a, dϕ_a = evalϕdϕ!(info, fdf!, a)
    # ϕ_b, dϕ_b = evalϕdϕ!(info, fdf!, b)
    # oppositeslopecondition(
    # ϕ_a,
    # dϕ_a,
    # dϕ_b,
    # ϕ_0,
    # ϵ,
    # )

    c = computesecant(a, b, dϕ_a, dϕ_b)

    A, B, status = updatebracketinverval!(
        info,
        config.fdf!,
        a, b, c, ϕ_0, ϕ_c, dϕ_c, ϵ;
        θ = config.θ,
        max_iters = config.max_iters,
    )

    ϕ_A, dϕ_A = evalϕdϕ!(info, config.fdf!, A)
    ϕ_B, dϕ_B = evalϕdϕ!(info, config.fdf!, B)
    
    # check if (A,B) needs to be further refined.

    need_further_update = false
    new_c = zero(T)
    if c == B
        #ϕ_B, dϕ_B = evalϕdϕ!(info, fdf!, B)
        new_c = computesecant(b, B, dϕ_b, dϕ_B)
        
        need_further_update = true

    elseif c == A
        #ϕ_A, dϕ_A = evalϕdϕ!(info, fdf!, A)
        new_c = computesecant(a, A, dϕ_a, dϕ_A)
        
        need_further_update = true

    end

    a_new = A
    b_new = B
    #ϕ_a_new, dϕ_a_new = ϕ_A, dϕ_A
    #ϕ_b_new, dϕ_b_new = ϕ_B, dϕ_B
    if need_further_update
        a_new, b_new, status = updatebracketinverval(A, B, new_c)

        a_new, b_new, status = updatebracketinverval!(
            info,
            config.fdf!,
            A, B, new_c, ϕ_0, ϕ_c, dϕ_c, ϵ;
            θ = config.θ,
            max_iters = config.max_iters,
        )
        #ϕ_a_new, dϕ_a_new = evalϕdϕ!(info, config.fdf!, a_new)
        #ϕ_b_new, dϕ_b_new = evalϕdϕ!(info, config.fdf!, b_new)
    end

    # if !(checkinterval(a, b, A, B, ϕ_0_plus_ϵ))
    #     return A, B, false
    # end
    
    #status = a_new < b_new
    return returninterval(a_new, b_new)#, ϕ_a_new, dϕ_a_new, ϕ_b_new, dϕ_b_new
end


############## initial interval. also use for reset.

# called [a,b] = bracket(c) on page 11/25 of Alg851.
# generate an interval that satisfies eqn 29 from the initial guess: [0,c].
function generateinterval(c::T; ρ = 0.1) where T

    c0 = c
    for j = 0
        #
        if dϕ(c_j) < 0

            if ϕ_cj > ϕ_0 + ϵ
                # B2.
                generate a, b via U3-c, with a_bar = 0, b_bar = cj
            else
                # B3.
                cj_p_1 = ρ*cj
                j += 1
                B1
            end
        else

            # B1
            b = cj
            find largest int i s.t. i<j,  ϕ(c_i) <= ϕ_0 + ϵ
            a = ci

            return a, b
        end

    end
end

function initialc()
    #
    
end

############### line search.

# eqn 27 of Alg851
function HZlinesearch()



end