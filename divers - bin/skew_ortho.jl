using LinearAlgebra

n = 6
A = randn(2n, 2n)
A = A - A'  # Make A skew-symmetric
SVD = svd(A)
P = SVD.U * SVD.Vt
Q = Matrix(qr(randn(2n, 2n)).Q)
J = [zeros(n, n) -I(n); I(n) zeros(n, n)]

SVD = svd(P[n+1:2n, 1:n])
U = SVD.U
for i∈2:2:n
    U[:, i] .= -U[:, i]
end
V = [SVD.V zeros(n, n); zeros(n,n)  U]
P2 = V' * P * V 
display(round.(V' * A * V, digits=2))
H = P2[n+1:2n, 1:n] + 1im * P2[1:n, 1:n]
S = eigen(Hermitian(H))

Q1 = S.vectors
M= [imag.(Q1) -real.(Q1); real.(Q1) imag.(Q1)]
for i = 1:n
    M[:, n+i] .*= sign(S.values[i])
end
P3 = M' * P2 * M

display(round.(M' * V' * A * V * M, digits = 2))
A2 = M' * V' * A * V * M
S2 = eigen(Hermitian(A2[n+1:2n, 1:n] + 1im * A2[1:n, 1:n]))
Q2 = S2.vectors
M2= [imag.(Q2) -real.(Q2); real.(Q2) imag.(Q2)]
for i = 1:n
    M2[:, n+i] .*= sign(S2.values[i])
end
display(round.(M2' * A2* M2, digits=2))

S = schur(P)
J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
kk = [1:2:2n; 2:2:2n]
Q = S.vectors[:, kk]
#display(round.(Q * (I - 1im * J) * Q', digits=2))
#display(round.(Q * (J + 1im * I) * Q', digits=2))

Q1 = Q[:, 1:n]
Q2 = Q[:, n+1:2n]
F = 1im * Q1 + Q2
display(round.(F * F', digits=2))
display(round.(P+1im *I, digits=2))
