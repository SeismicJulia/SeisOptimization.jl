@testset "non-mutating operator without keyward parameters" begin

    for Tv in [Float32, Float64, ComplexF32, ComplexF64]

        # size of the linear operator
        m = 33; n = 17;
        A = randn(Tv, m, n)

        # generate mutating forward operator
        function forward_generator(A)

            g = function (x)
                    y = A * x
                    return y
                end
            return g
        end
        f1 = forward_generator(A)

        # generate mutating adjoint operator
        function adjoint_generator(A)

            g = function (x)
                    y = A' * x
                    return y
                end
            return g
        end
        f2 = adjoint_generator(A)

        # create a linear operator
        L = LinearOperator(Tv, f1, f2, m, n)

        # apply forward operator to input x
        x = rand(Tv, n)
        y = Vector{Tv}(undef, m)
        y1= Vector{Tv}(undef, m)
        A_mul_b!(y, L, x)
        y1 = L * x
        y2 = A * x
        @test isapprox(y, y1)
        @test isapprox(y, y2)

        # apply adjoint operator to input x
        x = randn(Tv, m)
        y = Vector{Tv}(undef, n)
        Ac_mul_b!(y, L, x)
        y1 = A' * x
        @test isapprox(y, y1)

        # transpose opertor
        At_mul_b!(y, L, x)
        mul!(y1, transpose(A), x)
        @test isapprox(y, y1)

    end
end

@testset "non-mutating operator with keyword parameters" begin

    for Tv in [Float32, Float64, ComplexF32, ComplexF64]

        # size of the linear operator
        m = 33; n = 17;
        A = randn(Tv, m, n)

        # define forward operator which support mutating
        function forward_op(x; B=0)
            y = B * x
            return y
        end

        # define adjoint operator which support mutating
        function adjoint_op(x; B=0)
            y = B' * x
            return y
        end

        # create a linear operator
        L = LinearOperator(Tv, forward_op, adjoint_op, m, n; forward_params=(B=A,), adjoint_params=(B=A,))

        # apply forward operator to input x
        x = randn(Tv, n)
        y = Vector{Tv}(undef, m)
        y1= Vector{Tv}(undef, m)
        A_mul_b!(y, L, x)
        y1 = L * x
        y2 = A * x
        @test isapprox(y, y1)
        @test isapprox(y, y2)

        # apply adjoint operator to input x
        x = randn(Tv, m)
        y = Vector{Tv}(undef, n)
        Ac_mul_b!(y, L, x)
        y1 = A' * x
        @test isapprox(y, y1)

        # transpose opertor
        At_mul_b!(y, L, x)
        mul!(y1, transpose(A), x)
        @test isapprox(y, y1)

    end
end

@testset "mutating operator without keyward parameters" begin

    for Tv in [Float32, Float64, ComplexF32, ComplexF64]

        # size of the linear operator
        m = 127; n = 129;
        A = randn(Tv, m, n)

        # generate mutating forward operator
        function forward_generator(A)

            g = function (y, x)
                    mul!(y, A, x)
                    return nothing
                end
            return g
        end
        f1 = forward_generator(A)

        # generate mutating adjoint operator
        function adjoint_generator(A)

            g = function (y, x)
                    mul!(y, adjoint(A), x)
                    return nothing
                end
            return g
        end
        f2 = adjoint_generator(A)

        # create a linear operator
        L = LinearOperator(Tv, f1, f2, m, n)

        # apply forward operator to input x
        x = rand(Tv, n)
        y = Vector{Tv}(undef, m)
        y1= Vector{Tv}(undef, m)
        A_mul_b!(y, L, x)
        y1 = L * x
        y2 = A * x
        @test isapprox(y, y1)
        @test isapprox(y, y2)

        # apply adjoint operator to input x
        x = randn(Tv, m)
        y = Vector{Tv}(undef, n)
        Ac_mul_b!(y, L, x)
        y1 = A' * x
        @test isapprox(y, y1)

        # transpose opertor
        At_mul_b!(y, L, x)
        mul!(y1, transpose(A),x)
        @test isapprox(y, y1)

    end
end

@testset "mutating operator with keyward parameters" begin

    for Tv in [Float32, Float64, ComplexF32, ComplexF64]

        # size of the linear operator
        m = 100; n = 90;
        A = randn(Tv, m, n)

        # define forward operator which support mutating
        function forward_op!(y, x; B=0)
            mul!(y, B, x)
            return nothing
        end

        # define adjoint operator which support mutating
        function adjoint_op!(y, x; B=0)
            mul!(y, adjoint(B), x)
            return nothing
        end

        # create a linear operator
        L = LinearOperator(Tv, forward_op!, adjoint_op!, m, n; forward_params=(B=A,), adjoint_params=(B=A,))

        # apply forward operator to input x
        x = rand(Tv, n)
        y = Vector{Tv}(undef, m)
        y1= Vector{Tv}(undef, m)
        A_mul_b!(y, L, x)
        y1 = L * x
        y2 = A * x
        @test isapprox(y, y1)
        @test isapprox(y, y2)

        # apply adjoint operator to input x
        x = randn(Tv, m)
        y = Vector{Tv}(undef, n)
        Ac_mul_b!(y, L, x)
        y1 = A' * x
        @test isapprox(y, y1)

        # transpose opertor
        At_mul_b!(y, L, x)
        mul!(y1, transpose(A),x)
        @test isapprox(y, y1)

    end
end
