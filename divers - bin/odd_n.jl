using LinearAlgebra
include("hamiltoschur.jl")
#=
n = 4
nodd = 2 * n + 1
neven = 2 * n + 2
ε = eps(Float64)
#bad case
V = qr(randn(nodd, nodd)).Q
Σ = exp2.(-LinRange(0, 52, n))
SA = zeros(nodd, nodd)
S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
SA[1:2n, 1:2n] .= S
A = V * SA * V'
A = (A - A') / 2
B = zeros(neven, neven)
B[1:nodd, 1:nodd] .= A
E = randn(neven, neven)
#E[:, neven] .= randn(neven)
E = E - E'  # Make E skew-symmetric
E ./= norm(E)
E .*= ε
B .= B .+ E

Q, Σ = hamiltoschur!(B)
#display(round.(Q; digits=3))
display(norm(Q' * Q - I(neven))/sqrt(neven))
QA = zeros(nodd, nodd)


QA[:, 1:n] .= Q[1:nodd, 1:n]
QA[:, n+1:2n] .= Q[1:nodd, n+2:neven-1]
display(norm(Q[neven, 1:n]))
V = QA[:, 1:2n]
vkernel = randn(nodd)
vkernel .-= V * (V' * vkernel)
vkernel ./= norm(vkernel)
QA[:, nodd] .= vkernel
SA = zeros(nodd, nodd)
SA[1:2n, 1:2n] .= [zeros(n, n) Diagonal(-Σ[1:n]); Diagonal(Σ[1:n]) zeros(n, n)]
display(norm(A  * QA - QA * SA)/ norm(A))
display(norm(QA' * QA - I) / sqrt(2n))
=#
n = 6
nodd = 2 * n + 1
V = qr(randn(nodd, nodd)).Q
Σ = exp2.(-LinRange(0, 52, n))
SA = zeros(nodd, nodd)
S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
SA[1:2n, 1:2n] .= S
A = V * SA * V'
#A = randn(nodd, nodd)
A = (A - A') / 2

E = randn(nodd, nodd)
E = E - E'  # Make E skew-symmetric
E ./= norm(E)
E .*= eps(Float64) * norm(A)
A = A .+ E
P = polarqr!(copy(A))
x = randn(nodd)
z = x + P * (P * x)
z ./= norm(z)
V = randn(nodd, n)
V .-= z * (z' * V)
temp = zeros(nodd, n)
mul!(temp, P, V, 1, 0)
U = complex.(temp[:, 1:n], V)
M = Matrix(qr(U).Q)
#Use Q as temp
Q = zeros(nodd, n)
temp = Q[:, 1:n]
Mr = √2 .* real.(M)        #reduce allocations
Mi = -√2 .* imag.(M)
Z = [Mr Mi z]
display(Z' * P *Z)
display(Z' * A * Z)
Ah = Z' * A * Z
display(Ah[1:n, 1:n] - Ah[n+1:2n, n+1:2n])
display(Ah[n+1:2n, 1:n] + Ah[1:n, n+1:2n])
println("Done!")

#=
#Improve accuracy of z
        
        display(norm(A * z) / norm(A))
        #Arnoldi MGS
        nk = 5
        K = zeros(T, N, nk)
        K[:, 1] .= z
        for j in 1:nk-1
            mul!(K[:, j+1], A, K[:, j], 1, 0)
            for i ∈ 1:j-1
                K[:, j+1] .-= dot(K[:, i], K[:, j+1]) .* K[:, i]
            end
            K[:, j+1] ./= norm(K[:, j+1])
        end
        H2 = K' * A * K
        r = schur(H2).vectors[:, 1]
        #display(schur(H2).vectors)
        z = K * r
        z ./= norm(z)
        display(norm(A * z) / norm(A))
        =#