# generic barrier algorithm.

# the general case. sec 11.2.1 (Boyd 2004).
# in standard form, i.e. h[i](x) <= 0 is one of the inequality constraints.
function evallogbarrierandderivatives(
    h_x_set::Vector{T},
    dh_x_set::Vector{Vector{T}},
    hdh_set!::Vector,
    x::Vector{T},
    ) where T

    @assert length(h_x_set) == length(dh_x_set) == length(hdh_set!)
    @assert !isempty(dhd_set!) # must have at least one constraint. Otherwise don't call this function.

    hdh_set!(h_x_set, dh_x_set, x)

    ϕ_x = -sum( log(-h_x_set[m]) for m in eachindex(h_x_set))
end

# note the use of a barrier function NEEDS all line search functions or any function evals to never violate the constraint.