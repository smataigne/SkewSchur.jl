using LinearAlgebra, SkewLinearAlgebra
include("polar.jl")

"""
    hamiltoschur!(A::AbstractMatrix{T}; random = true, polar = :qr) where T
Compute the real Schur decomposition of a real skew-symmetric matrix `A` of size `2n x 2n` using a skew-symmetric Hamiltonian structure.

Returns an orthogonal matrix `Q` and a diagonal matrix `Σ` such that `A = Q * [0 -Σ; Σ 0] * Q'`.

If `polar` is `:qr`, `:chol`, `:hybrid` or `:svd`, the polar decomposition is computed using the QR-based, Cholesky-based, hybrid or SVD-based method, respectively.
"""
@views function hamiltoschur!(A::AbstractMatrix{T::TypeVar}; polar = :qr) where T
    N = size(A, 1)
    ε = eps(T)
    n = N ÷ 2
    Q = zeros(T, N, N)
    H = zeros(T, n, n)
    Ω = zeros(T, n, n)
    temp = zeros(T, N, n)
    V = randn(T, N, n)
    x = zeros(T, N)
    z = randn(T, N)

    #We add a skew-symmetric perturbation for stability
    E = randn(T, N, N)
    E = E - E'
    E .*= (ε * norm(A)) / norm(E)
    A .+= E
    #In case of odd-sized matrix, enforce the use of the QR-based polar decomposition
    if isodd(N) && polar != :qr 
            @warn "Matrix is odd-sized, using QR-based polar decomposition."
            polar = :qr
    end

    if polar == :qr
        P = polarqr!(copy(A))
    elseif polar == :chol
        P = polarchol!(copy(A))
    elseif polar == :hybrid
        P = polarhybrid!(copy(A))
    elseif polar == :newton
        P = polarnewtonschulz!(copy(A))
    else
        P = polarsvd!(copy(A))
    end

    #If the matrix is odd-sized, find the real eigenvector z corresponding to the eigenvalue 0 of A.
    if isodd(N)
        mul!(x, P, z, 1, 0)
        mul!(z, P, x, 1, 1)
        z ./= norm(z)
        Q[:, N] .= z
        mul!(x[1:n], V', z, 1, 0)
        BLAS.ger!(-1.0, z, x[1:n], V)
    end
    
    mul!(temp, P, V, 1, 0)
    U = complex.(temp[:, 1:n], V)
    M = Matrix(qr(U).Q)
    #Use Q as temp
    temp = Q[:, 1:n]
    Mr = real.(M)        #reduce allocations
    Mi = imag.(M)
    sqrt2 = √2
    #reduce allocations
    mul!(temp, A, Mr , sqrt2, 0)
    mul!(Ω, Mr', temp, sqrt2, 0)
    mul!(H, Mi', temp, sqrt2, 0)
    H2 = complex.(H, Ω)
    hermitianpart!(H)
    E = eigen!(Hermitian(H2))
    H .= real.(E.vectors)    #reduce allocations
    Ω .= imag.(E.vectors)
    mul!(Q[:, 1:n], Mr, Ω, sqrt2, 0)
    mul!(Q[:, 1:n], Mi, H, sqrt2, 1)
    mul!(Q[:, n+1:2n], Mr, H, -sqrt2, 0)
    mul!(Q[:, n+1:2n], Mi, Ω, sqrt2, 1)
    for k ∈ 1:n
        Q[:, n + k] .*= sign(E.values[k])
    end
    return Q, Diagonal(abs.(E.values))
end

"""
    hamiltoschur(A::AbstractMatrix{T}; random = true, polar = :qr) where T
Compute the real Schur decomposition of a real skew-symmetric matrix `A` of size `2n x 2n` using a skew-symmetric Hamiltonian structure.

Returns an orthogonal matrix `Q` and a diagonal matrix `Σ` such that `A = Q * [0 -Σ; Σ 0] * Q'`.

If `polar` is `:qr`, `:chol`, `:hybrid` or `:svd`, the polar decomposition is computed using the QR-based, Cholesky-based, hybrid or SVD-based method, respectively.
"""
hamiltoschur(A::AbstractMatrix{T}; polar = :qr) where T = hamiltoschur!(copy(A); polar = polar)
