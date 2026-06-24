using LinearAlgebra, BenchmarkTools, SkewLinearAlgebra
include("hamiltoschur.jl")

BLAS.set_num_threads(1)
for i ∈1:10
    n = 4
    A = randn(2n, 2n)
    A = A - A'

    #@btime hamiltoschur!(copy(A); safe = false);
    X = polarqr!(copy(A))
    display(tr(X'A))
end
#=
Q, v = hamiltoschur!(copy(A); random = true, polar=:qr)

@btime schur(A)
@btime eigen(Hermitian(1im * A))
=#
print("Done\n")