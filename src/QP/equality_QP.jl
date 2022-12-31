# 16.1, (Nocedal 2006).
struct EqualityQP{T}
    G::Matrix{T}
    c::Vector{T}
    A::Matrix{T}
    b::Vector{T}
end

# returns empty matrix if size(A,1) >= size(A,2)
function getnullspacebasis(A::Matrix{T})::Matrix{T} where T
    m,n = size(A)
    _,_,Vt = svd(A, full = true)
    
    return Vt[:,m+1:end]
end

# equation 15.22 (Nocedal 2006).
# returns empty matrix if A is invertible.
function getYZmatQP(A::Matrix{T})::Tuple{Matrix{T},Matrix{T}} where T
    m,n = size(A)
    @assert m <= n

    Q,R = qr(A')
    Q1 = Q[:,1:m] # Y
    Q2 = Q[:,m+1:end] # Z

    return Q1, Q2
end

function solveequalityQP(config::EqualityQP{T}) where T

    G, c, A, b = config.G, config.c, config.A, config.b

    m,n = size(A)
    @assert m <= n

    # eqn 16.5: first-order necessary conditions.
    #g = c + G*x
    #h = A*x - b
    #KKT_mat = [G A'; A zeros(T,size(A'))]
    #KKT_vec = [g; h]

    # eqn 16.22, via reduced system.
    #Z = getnullspacebasis(A)
    Y,Z = getYZmatQP(A)

    if !isposdef(Z'*G*Z)
        println("Z'GZ is not posdef.")
        return zeros(T,n), false
    end

    x_y = (A*Y)\b
    c_z = Z'*(G*Y*x_y + c)
    x_z = Z'*G*Z\(-c_z) # eqn 16.24.

    x_star = Y*x_y + Z*x_z
    
    return x_star, true
end