using Test
import FiniteDifferences
using LinearAlgebra

include("../examples/helpers/test_funcs.jl")

@testset "test function derivative implementation" begin
    
    N_tests = 10
    zero_tol = 1e-12
    ND_zero_tol = 1e-5

    for _ = 1:N_tests
        
        fdf! = boothfdf! # booth test func.

        # double check if the solution has gradient norm of zero.
        x_oracle = [1.0; 3.0]
        df_x = ones(length(x_oracle))
        f_x_oracle = fdf!(df_x, x_oracle)
        @test norm(df_x) < zero_tol

        x_test = randn(2)
        f_x = fdf!(df_x, x_test)
        @show f_x, df_x

        ND_accuracy_order = 8

        objectivefunc = xx->fdf!(df_x, xx)
        df_ND = pp->FiniteDifferences.grad(
            FiniteDifferences.central_fdm(
                ND_accuracy_order,
                1,
            ),
            objectivefunc,
            pp
        )[1]

        df_ND_eval = df_ND(x_test)

        f_x = fdf!(df_x, x_test)
        @test norm(df_x-df_ND_eval) < ND_zero_tol # passed.
    end
end