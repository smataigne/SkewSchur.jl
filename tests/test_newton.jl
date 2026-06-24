using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools
include("polar.jl")
#=
n = 10
Q = Matrix(qr(randn(2n, 2n)).Q)
Σ = Diagonal(rand(n))
J = [zeros(n, n) -Σ; Σ zeros(n, n)]
A = Q * J * Q'
A = (A - A') / 2
BLAS.set_num_threads(10)
polarnewtonschulz!(copy(A))
polarchol!(copy(A))
n = 3000
Q = Matrix(qr(randn(2n, 2n)).Q)
Σ = Diagonal(rand(n))
J = [zeros(n, n) -Σ; Σ zeros(n, n)]
A = Q * J * Q'
A = (A - A') / 2
@time polarnewtonschulz!(copy(A))
@time polarchol!(copy(A))
print("Done\n")
print("Done\n")
=#
n = 501
A = randn(n, n)
A = (A - A') / 2
A2 = randn(n+1,n+1) * eps(Float64)
A2  = (A2 - A2') / 2
A2[1:n, 1:n] .= A
P = polarqr!(copy(A2))