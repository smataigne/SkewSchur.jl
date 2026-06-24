using LinearAlgebra, Test

@views function skeworthoschur(A::AbstractMatrix{T}) where T
    n = size(A, 1) ÷ 2
    Q = zeros(T, 2n, 2n)
    V = randn(T, 2n, n)
    temp = zeros(T, 2n, n)
    mul!(temp, A, V, 1, 0)
    U = complex.(temp[:, 1:n], V)
    M = Matrix(qr(U).Q)
    Q[:, 1:n] .= real.(M)
    Q[:, n+1:2n] .= -imag.(M)
    Q .*= sqrt(2)
    return Q
end
#=
for n ∈ [10, 100, 1000]
    ε = 20 * sqrt(n) * eps(eltype(Float64))
    Q = Matrix(qr(randn(2n, 2n)).Q)
    J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
    A = -Q * J * Q'
    Q2 = skeworthoschur(A)
    display(@test norm(A * Q2 - Q2 * J) / sqrt(2n) < ε)
end
=#
#=
n = 3
Q = Matrix(qr(randn(2n, 2n)).Q)
J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
A = Q * J * Q'
rank(A + 1im * I(2n))
qr(J + 1im * I(2n))
Q1 = Q[:, 1:n]
Q2 = Q[:, n+1:2n]
V = complex.(Q1, Q2) / √2
display(norm(V'V - I(n)))
M = [real.(V) -imag.(V)]
display(Q'M)
=#

