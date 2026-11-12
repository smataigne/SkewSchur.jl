using LinearAlgebra, Plots, BenchmarkTools, LaTeXStrings
include("../src/hamiltoschur.jl")
include("../src/other_schur.jl")

BLAS.set_num_threads(32)
n = 4000

Q = Matrix(qr(randn(2n, 2n)).Q)
D = Diagonal(1.0 .+ 0.5*rand(n))
S = [zeros(n, n) -D; D zeros(n, n)]
A = Q * S * Q'

#=
A = randn(2n, 2n)
A = (A - A') / 2
A ./= norm(A)
=#
#times_hamiltoschur_svd=  @elapsed hamiltoschur!(copy(A); polar= :svd)
times_hamiltoschur_hybrid=  @elapsed hamiltoschur!(copy(A); polar= :newton)
#times_hamiltoschur_halley=  @elapsed hamiltoschur!(copy(A); polar= :halley)
#times_householder_schur= @elapsed householder_schur(copy(A))
times_hermitian_eigen= @elapsed hermitian_eigen(copy(A))
#times_lapack_schur=  @elapsed schur(copy(A))

#display("Time for Algorithm 4.1 with PSVD: $times_hamiltoschur_svd")
display("Time for Algorithm 4.1 with PHybrid: $times_hamiltoschur_hybrid")
#display("Time for Algorithm 4.1 with PHalley: $times_hamiltoschur_halley")
#display("Time for Algorithm 4.1 with PHouseholder: $times_householder_schur")
display("Time for Algorithm 4.1 with PHermitian: $times_hermitian_eigen")
#display("Time for Algorithm 4.1 with PLapack: $times_lapack_schur")