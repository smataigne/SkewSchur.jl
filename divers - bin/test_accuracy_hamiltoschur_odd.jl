using LinearAlgebra
include("../src/hamiltoschur.jl")

N = 111
n = N ÷ 2
A = randn(N, N)
A = A - A'  # Make A skew-symmetric
Q, v = hamiltoschur!(A)
Σ = Diagonal(v)
S = zeros(N, N)
S[n+1:2n, 1:n] .= Σ
S[1:n, n+1:2n] .= -Σ
display(norm(A * Q - Q * S) / norm(A))

