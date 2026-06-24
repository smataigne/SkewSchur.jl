using LinearAlgebra, BenchmarkTools

@views function gramschmidt!(Q::AbstractMatrix)
    n = size(Q, 2)
    for j ∈ 1:n
        for k ∈ 1:j-1
            r = -dot(Q[:, k], Q[:, j])
            axpy!(r, Q[:, k], Q[:, j])
        end
        Q[:, j] ./= norm(Q[:, j])
    end
end

BLAS.set_num_threads(6)
n = 4000
p = n ÷ 2
Q = randn(n, p)
@btime Matrix(qr(Q).Q)
@btime gramschmidt!(copy(Q))