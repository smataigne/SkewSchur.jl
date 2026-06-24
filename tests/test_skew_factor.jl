using LinearAlgebra

n = 4
S = Matrix(qr(randn(2n, 2n)).Q)
if det(S) < 0
    S[:,1] .= -S[:,1]
end
A = S - S' # Make A skew-symmetric

SVD = svd(A)
P = SVD.U * SVD.Vt
O = (P + 1im * I(2n)) * [I(n); zeros(n,n)]
Q = Matrix(qr(O).Q)
display(Q'O)
#Q = (P + [1im * I(n) zeros(n,n); zeros(n,n) 1im * I(n)])/sqrt(2) * [I(n); zeros(n,n)]
#Q = (P + 1im * I(2n)) * [I(n); zeros(n,n)]
#display(Q'Q)


Q = [real.(Q) imag.(Q)] * sqrt(2)
display(Q)

# Verify that Q' * A * Q is skew-symmetric
B = Q' * A * Q
#display(Q)
display(B)

M = A[1:n, n+1:2n]
SVD = svd(M)
P = SVD.U * SVD.Vt
A[:, 1:n] .= A[:, 1:n] * P
A[1:n, :] .= P' * A[1:n, :]
display(A)


#=
Q = Matrix(qr(randn(ComplexF64, n, n)).Q)
V = sqrt(2) * [real.(Q) imag.(Q); -imag.(Q) real.(Q)]
Σ = Diagonal(randn(n))
A = V * [zeros(n, n) Σ; -Σ zeros(n, n)] * V'
display(A)
=#

A = randn(2n, 2n)
A = A - A'  # Make A skew-symmetric
SVD = svd(A)
P = SVD.U * SVD.Vt
V = Matrix(qr(randn(2n,2n)).Q)
M = svd(P[n+1:2n, 1:n])
G = [M.V zeros(n, n); zeros(n, n) M.U]
P2 = G' * P * G
K = P2[n+1:2n, 1:n] + 1im * P2[1:n,1:n]
S = eigen(K)
Q1 = S.vectors
M= [imag.(Q1) real.(Q1); -real.(Q1) imag.(Q1)]
#M= [real.(Q1) -imag.(Q1); imag.(Q1) real.(Q1)]
display(M' * P2 * M)
V = P * V
#Q = Matrix(S.Q)
Q = randn(2n, 2n)
B = G' * A * G
kk = [1:2:2n; 2:2:2n]
println("Blocks:")
display(B[kk, kk])

A = randn(2n, 2n)
A = A - A'  # Make A skew-symmetric
A[1:n, n+1:2n] .= Matrix(Diagonal(diag(A[1:n, n+1:2n] + A[1:n, n+1:2n]')))
A[n+1:2n, 1:n] .= -A[1:n, n+1:2n]'

#display(A)
#H = A[1:n, n+1:2n] + A[n+1:2n, 1:n]

Ω1 = A[1:n, 1:n]
Ω2 = A[n+1:2n, n+1:2n]
B = Ω1 - Ω2
B2 = B #+ 1im * H 
S = eigen(B2)
Q = Matrix(S.vectors)
V = [real.(Q); imag.(Q)]
#display(V' * [B zeros(n, n); zeros(n, n) -B] * V)
J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
R = [V J*V]
A = R' * A * R
println("Ok")
#display(A)

#=
B = A[1:n, n+1:2n] + A[n+1:2n, 1:n]
B2 = B #+ 1im * H 
S = eigen(B2)
Q = Matrix(S.vectors)
V = [imag.(Q); real.(Q)]
#display(V' * [B zeros(n, n); zeros(n, n) -B] * V)
J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
R = [V J*V]
A = R' * A * R

A = randn(2n, 2n)
A = A - A'  # Make A skew-symmetric
Ω1 = A[1:n, 1:n]
Ω2 = A[n+1:2n, n+1:2n]
M = A[n+1:2n, 1:n]
B = 1im* (M-M') + (Ω1 - Ω2)
S = schur(B)
Q = Matrix(S.vectors)
V = [real.(Q); imag.(Q)]
#display(V' * [B zeros(n, n); zeros(n, n) -B] * V)
J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
R = [V J*V]
K = [Ω1 - Ω2 M-M'; M-M' -Ω2 + Ω1]
S2 = schur(K)
Q2 = Matrix(S2.vectors)[:, kk]

display(K)
=#