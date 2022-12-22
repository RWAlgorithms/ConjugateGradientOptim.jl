
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
function oppositeslopecondition(ϕ, dϕ, a::T, b::T, ϵ) where T <: AbstractFloat
    ϕ(a) <= ϕ(0) + ϵ
    dϕ(a) < 0
    dϕ(b) >= 0
end

# error-handling
function returninterval(a::T, b::T)::Tuple{T,T,Bool} where T
    if !(a < b)
        return a, b, false
    end

    return a, b, true
end

# [a,b] is the current bracketing interval, c is an update point.
# This is 11/25, interval update rules, in Alg851.
function updatebracketinverval(
    ϕ,
    dϕ,
    a::T,
    b::T,
    c::T;
    θ = convert(T,0.5),
    max_iters::Integer = 1000,
    ) where T <: AbstractFloat
    
    # U0.
    if !(a < c < b)
        return returninterval(a, b)
    end

    if dϕ(c) < 0
        if ϕ(c) > (ϕ(0) + ϵ)
            # U3.
            out_a = a
            out_b = b

            for n = 1:max_iters
                
                d = (1-θ)*out_a + θ*out_b
                if dϕ(d) < 0
                
                    if ϕ(d) > ϕ(0) + ϵ
                        # U3c.
                        out_b = d
                    else
                        # U3b.
                        out_a = d
                    end

                else
                     # U3a.
                     out_b = d
                     return returninterval(out_a, out_b)
                end
            end

            return a, b, false # line search reached maximum iterations, consider as a failed search.
        else
            
            # U2.
            return returninterval(c, b)
        end
    end

    # U1.
    return returninterval(a, c)
end

function checkinterval(a, b, a_new, b_new)
    if !(a <= a_new < b_new <= b)
        return false
    end

    # these should be satisfied.
    dϕ(a_new) < 0 
    ϕ(a_new) <= ϕ(0) + ϵ

    dϕ(b_new) < 0 
    ϕ(b_new) > ϕ(0) + ϵ
    
end

################ secant update as candidate for a bracket end point.
# pg 11/25 of Alg851

function computesecant(a::T, b::T, dϕ_a::T, dϕ_b::T)::T where T
    return a*dϕ_b - b*dϕ_a / (dϕ_b - dϕ_a)
end
    
# titled secant in Alg851.
function getintervalcandidate(a::T, b::T, dϕ_a::T, dϕ_b::T)::T where T

    c = computesecant(a, b, dϕ_a, dϕ_b)
    A, B, status = updatebracketinverval()
    if !(checkinterval(a, b, A, B))
        return A, B, false
    end

    need_further_update = false
    new_c = zero(T)
    if c == B
        new_c = computesecant(b, B)
        
        need_further_update = true

    elseif c == A
        new_c = computesecant(a, A)
        
        need_further_update = true

    end

    a_new = A
    b_new = B
    if need_further_update
        a_new, b_new, status = updatebracketinverval(A, B, new_c)
    end
    
    return returninterval(a_new, b_new)
end

#I am here. solidify this, but don't convert to fdf!(), but use pre-computed values as much as possible up to this point.

############### line search.

# eqn 27 of Alg851
function HZlinesearch()



end