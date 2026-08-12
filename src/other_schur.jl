using LinearAlgebra, SkewLinearAlgebra

@views function even_odd_perm!(A::AbstractMatrix)
    n = size(A, 1)
    Base.permutecols!!(A, [1:2:n;2:2:n])
end

"""
    householder_schur(A::AbstractMatrix{T}) where T
Compute the Schur decomposition of a skew-symmetric matrix `A` of size `2n x `2n` using a tridiagonalization approach. Returns an orthogonal matrix `V` and a diagonal matrix `Σ` such that `A = V * [0 -Σ; Σ 0] * V'`.
"""
@views function householder_schur(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    n2 = n ÷ 2
    n2b = n2 + Int(isodd(n))
    V = similar(A, n, n)
    Ω = skewhermitian(A)
    H = hessenberg(Ω)
    K = Matrix(H.Q) #"K" for "Krylov" basis
    β = H.H.ev
    even_odd_perm!(K)
    β[2:2:end] .*= (-1)               #Abstract even-odd permutation of β to bidiagonal form.
    l = length(β) - Int(isodd(n))
    B = Bidiagonal(Vector(β[1:2:l]), Vector(β[2:2:l]), :U)
    SVD = svd!(B)
    mul!(V[:, 1:n2], K[:, 1:n2], SVD.Vt', 1, 0)
    mul!(V[:, (n2b+1):end], K[:, (n2b+1):end], SVD.U, 1, 0)
    Σ = Diagonal(SVD.S)
    return V, Σ
end

"""
    hermitian_eigen(A::AbstractMatrix)
Compute the eigenvalue decomposition of a skew-symmetric matrix `A` from that of the corresponding Hermitian matrix `1im * A`. Returns a unitary matrix `U` and a diagonal matrix `Λ` such that `A = U * Λ * U'`.
"""
function hermitian_eigen(A::AbstractMatrix) 
    E = eigen(Hermitian(1im * A))
    Λ = Diagonal(complex.(0, -E.values))
    return E.vectors, Λ
end

"""
    randdiag(A::AbstractMatrix)
Compute a random diagonalization of a matrix `A` using a random linear combination of the Hermitian and skew-Hermitian parts of `A`. Returns a unitary matrix `U` and a diagonal matrix `D` such that `A = U * D * U'`.

Method from ``H. He and D. Kressner, A simple, randomized algorithm for diagonalizing normal matrices, Calcolo, 62 (2025), https://doi.org/10.1007/s10092-025-00654-z``
"""
@views function randdiag(A::AbstractMatrix)
    n = size(A, 1)
    μ₁ = randn()
    μ₂ = randn()
    H = hermitianpart(A)
    S = skewhermitian(A)
    M = Hermitian(complex.(μ₁ * H, μ₂ * S))
    E = eigen(M)
    λ = zeros(typeof(M[1, 1]), n)
    B = A * E.vectors
    for i ∈ 1:n
        λ[i] = dot(E.vectors[:, i], B[:, i])
    end
    return E.vectors, Diagonal(λ)
end

