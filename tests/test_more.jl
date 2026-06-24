using LinearAlgebra
#=
n = 10

#=
A = randn(2n, 2n)
A = (A - A') / 2  # Make A skew-symmetric

v = randn(2n)
M = zeros(2n, 2n)
M[:, 1] .= v
@views for i ∈ 2:2n
    mul!(M[:, i], A, M[:, i - 1])
    M[:, i] ./= norm(M[:, i])
end 
M = Matrix(qr(M[:, [1:2:end; 2:2:end]]).Q)
V = M[:, 1:n] + 1im * M[:, (n+1):2n]
V = Matrix(qr(V).Q)
V2 = [real.(V) -imag.(V)]
display(V2'*A*V2)
=#
A = tril(randn(2n, 2n))
A = A[2n:-1:1, :]
A = A - A'
display(sqrt(abs(det(A))))
B = A[n+1:2n, 1:n]
B = B[end:-1:1, :]
display(abs(det(B)))
=#

N = 1001
σ = 0
for k = 1:2:N
    global σ
    σ += k^2
end
display(σ * 6 / (N^3))
