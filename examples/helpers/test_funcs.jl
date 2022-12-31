
# f = xx->((xx[1]+2*xx[2]-7)^2+(2*x[1]+x[2]-5)^2)
function boothfdf!(g_p::Vector{T}, p::Vector{T}) where T <: AbstractFloat
    x, y = p

    f_p = (x+2*y-7)^2 + (2*x+y-5)^2

    g_p[begin] = 2*(x+2*y-7) + 2*(2*x+y-5)*2
    g_p[begin+1] = 2*(x+2*y-7)*2 + 2*(2*x+y-5)

    return f_p
end



################ legacy.

# M2 Mahfoud 1995 Niching Genetic Algorithms
# minimum at 0.1. other modews at 0.3 0.5 0.7 0.9
function Mahfoud2func(x)
    @assert 0.0 <= x[1] <= 1.0
    return 1.0-sin(5*π*x[1])^6*exp(-2.0*log(2)*((x[1]-0.1)/0.8)^2)+randn();
end

"""
```
rastriginfunc(x::Vector{T})::T where T <: AbstractFloat
```

Unconstrained minimum at `x_star = zeros(length(x))`.
"""
function rastriginfunc(x::Vector{T})::T where T <: AbstractFloat
    N=length(x)

    running_sum::Float64=0.0
    for i in eachindex(x)
        running_sum += x[i]*x[i]-10.0*cos(2*π*x[i])
    end

    return running_sum + 10*N
end

"""
```
rosenbrockfunc(x::Vector{T})::T where T <: AbstractFloat
```

Unconstrained minimum at `x_star = ones(length(x))`.
"""
function rosenbrockfunc(x::Vector{T})::T where T <: AbstractFloat
    d=length(x);
    running_sum=0.0;
    for i=1:d-1
        running_sum += (1-x[i])^2+100*(x[i+1]-x[i]^2)^2; 
    end
    return running_sum
end

function dejongsfunc(x)
    return sum(x.^2)
end

# f(x*)=-418.9829*d, x*=420.9687*ones
"""
```
rosenbrockfunc(x::Vector{T})::T where T <: AbstractFloat
```

Evaluates to zero at minimum.
Unconstrained minimum at `x_star = 420.9687 .* ones(length(x))`.
"""
function schwefelfunc(x::Vector{T})::T where T <: AbstractFloat

    running_sum = sum(
        x[i]*sin(sqrt(abs(x[i])))
        for i in eachindex(x)
    )

    return 418.9829*length(x) - running_sum
end



# mean-squared error of linear regression
function mseobjfunc(yn::Float64, ϕn::Vector{Float64}, θn::Vector{Float64})
    @assert length(ϕn) == length(θn)

    err = (yn - dot(ϕn,θn))^2
    @assert isfinite(err)

    return err
end


########## equality constrained QP example problems.

function equalityQPexample169()
    G = [6 2 1; 2 5 2; 1 2 4] .* 1.0
    c = [-8; -3; -3] .* 1.0
    A = [1 0 1; 0 1 1] .* 1.0
    b = [3; 0] .* 1.0

    x_oracle = [2; -1; 1] .* 1.0
    λ_oracle = [3; -2] .* 1.0

    return G, c, A, b, x_oracle, λ_oracle
end