

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