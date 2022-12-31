


###### Alg 2.2 in (Yuan 2019).

function minimizeobjective(
    fdf!,
    x_initial::Vector{T},
    config::CGConfig{T,BT,ET},
    linesearch_config::LineSearchConfig,
    ) where {T <: AbstractFloat, BT <: βConfig, ET}

    # ## parse.
    D = length(x_initial)
    max_iters = config.max_iters
    β_config = config.β_config
    #linesearch_config = config.linesearch_config

    # ## allocate.
    df_x = Vector{T}(undef, D)
    x = copy(x_initial)
    #u_prev = Vector{T}(undef, D)

    # ## Step 1 (Yuan 2019): initialize or allocate for first iteration.
    f_x::T = fdf!(df_x, x)
    norm_df_x::T = norm(df_x)
    norm_df_xp::T = NaN
    #β::T = zero(T)
    β = initializeβ(T, β_config)
    fdf_evals_ran::Int = -1
    f_x0 = f_x # objective of initial iterate.
    
    # ## return container.
    ret = Results(
        f_x,
        x,
        df_x,
        0,
        :incomplete,
        setuptrace(T, config.trace_status),
    )
    resizetrace!(ret.trace, max_iters)

    # ## line search.
    info = LineSearchContainer(T, D)
    initializeLineSearchContainer!(info, β_config, df_x, x)
    a_initial = NaN # use default value on first try. Use HagerZhang 851 algorithm later.

    # # Run algortihm.
    for n = 1:max_iters

        # check stopping conditions.
        if isfinite(f_x) && isfinite(norm_df_x)
            if norm_df_x < config.ϵ
                
                if f_x <= f_x0
                    # the previous iteration meets the stopping criteria. Return it.
                    updateresult!(
                        ret,
                        x,
                        df_x,
                        f_x,
                        n-1,
                        :success,
                    )
                    return ret
                end

                # the previous iteration meets the stopping criteria. However, got a larger objective than the initial iterate.
                updateresult!(
                    ret,
                    x,
                    df_x,
                    f_x,
                    n-1,
                    :increasing_objective,
                )
                return ret
            end
        end

        # step 3 (Yuan 2019): linesearch.
        f_xp, a_star, fdf_evals_ran, status_symbol = linesearch!(
            info,
            linesearch_config,
            fdf!,
            f_x,
            df_x,
            a_initial,
        )
        
        a_initial = a_star # set to last solved step.
        if status_symbol != :success
            # return the last known good iterate.
            updateresult!(
                ret,
                x,
                df_x,
                f_x,
                n-1,
                status_symbol,
            )
            return ret
        end

        # check if we have a numerically valid proposed iterate and its gradient, before updating the current iterate.
        norm_df_xp = norm(info.df_xp)
        if !isfinite(f_xp) || !isfinite(norm_df_xp)
            
            # terminate on indeterminant or overflowed derivatives even if the objective is valid.
            # this might happen for when the df_x is very large in the neighbourhood of the iterate x, which is common in primal barrier objectives.
            updateresult!(
                ret,
                x,
                df_x,
                f_x,
                n-1,
                :non_finite_objective_or_gradient_proposed,
            )
            return ret
        end
        
        # save some quantities for initial step a_initial.
        #f_x_prev = f_x
        #dϕ_0 = dot(df_x, info.u)

        # safe to update current iterate.
        # step 4 & 5 (Yuan 2019): update iterate and objective-related evaluations.
        #f_x = fdf!(info.df_xp, x) # temporarily use info.df_xp to store the gradient of the next iterate.
        β = getβ(
            β_config,
            info.df_xp,
            df_x,
            info.u,            
        )
        x[:] = info.xp

        f_x = f_xp
        df_x[:] = info.df_xp # now, safe to overwrite gradient of the iterate.
        info.x[:] = x
        norm_df_x = norm_df_xp
        
        # step 5: update search direction for next iteration.
        
        updatedir!(info.u, df_x, β)

        # seems slower than just simply set step to the last solved step.
        #a_initial = 2*(f_x-f_x_prev)/dot(df_x, info.u) # eqn 3.6 from (Nocedal 2006).
        #a_initial = a_star * dϕ_0/dot(df_x, info.u)

        # update trace.
        updatetrace!(
            ret.trace,
            f_x,
            norm_df_x,
            a_star,
            fdf_evals_ran,
            n,
        )
    end

    updateresult!(
        ret,
        x,
        df_x,
        f_x,
        max_iters,
        :max_iters_reached,
    )
    return ret
end

function minimizeobjectivererun(
    fdf!,
    x_initial::Vector{T},
    config::CGConfig{T,BT,ET},
    #linesearch_config::StrongWolfeBisection{T},
    linesearch_config::LineSearchConfig,
    rerun_config_tuples...
    )::Vector{Results{T,TraceContainer{T,ET}}} where {T <: AbstractFloat, BT <: βConfig, ET}
    
    rets = Vector{Results{T,TraceContainer{T,ET}}}(undef, 1)
    rets[begin] = ConjugateGradientOptim.minimizeobjective(
        fdf!,
        x_initial,
        config,
        linesearch_config,
    )

    # run with backup configs if unsuccessful.
    for k in eachindex(rerun_config_tuples)
        if rets[end].status != :success
            rerun_config, backup_linesearch_config = rerun_config_tuples[k]

            ret = ConjugateGradientOptim.minimizeobjective(
                fdf!,
                rets[end].minimizer,
                rerun_config,
                backup_linesearch_config,
            )
            push!(rets, ret)
        else
            return rets
        end
    end

    return rets
end