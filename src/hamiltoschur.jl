using LinearAlgebra, SkewLinearAlgebra
include("polar.jl")

"""
    hamiltoschur!(A::AbstractMatrix{T}; random = true, polar = :qr) where T
Compute the real Schur decomposition of a real skew-symmetric matrix `A` of size `2n x 2n` using a skew-symmetric Hamiltonian structure.

Returns an orthogonal matrix `Q` and a diagonal matrix `Σ` such that `A = Q * [0 -Σ; Σ 0] * Q'`.

If `random` is true, a random matrix is used to initialize the computation.

If `polar` is `:qr`, `:chol`, or `:svd`, the polar decomposition is computed using the QR-based, Cholesky-based, or SVD-based method, respectively.
"""
@views function hamiltoschur!(A::AbstractMatrix{T}; random = true, polar = :qr) where T
    n  = size(A, 1) ÷ 2
    Q = zeros(2n, 2n)
    H = zeros(T, n, n)
    Ω = zeros(T, n, n)
    if random
        if polar == :qr
            P = polarqr!(copy(A))
        elseif polar == :chol
            P = polarchol!(copy(A))
        elseif polar == :hybrid
            P = polarhybrid!(copy(A))
        else
            P = polarsvd!(copy(A))
        end
        V = randn(T, 2n, n)
        temp = zeros(2n, n)
        mul!(temp, P, V, 1, 0)
        U = complex.(temp[:, 1:n], V)
    else
        if polar == :qr
            U = complex.(polarqr!(copy(A))[:, 1:n], Matrix(one(T) * I, 2n, n))
        elseif polar == :chol
            U = complex.(polarchol!(copy(A))[:, 1:n], Matrix(one(T) * I, 2n, n))
        elseif polar == :hybrid
            U = complex.(polarhybrid!(copy(A))[:, 1:n], Matrix(one(T) * I, 2n, n))
        else
            U = complex.(polarsvd!(copy(A))[:, 1:n], Matrix(one(T) * I, 2n, n))
        end 
    end
    M = Matrix(qr(U).Q)
    #Use Q as temp
    temp = Q[:, 1:n]
    Mr = real.(M)        #reduce allocations
    Mi = imag.(M)
     #reduce allocations
    mul!(temp, A, Mr , sqrt(2), 0)
    mul!(Ω, Mr', temp, sqrt(2), 0)
    mul!(H, Mi', temp, sqrt(2), 0)
    H2 = complex.(H, Ω)
    hermitianpart!(H)
    E = eigen!(Hermitian(H2))
    H .= real.(E.vectors)    #reduce allocations
    Ω .= imag.(E.vectors)
    mul!(Q[:, 1:n], Mr, Ω, sqrt(2), 0)
    mul!(Q[:, 1:n], Mi, H, sqrt(2), 1)
    mul!(Q[:, n+1:2n], Mr, H, -sqrt(2), 0)
    mul!(Q[:, n+1:2n], Mi, Ω, sqrt(2), 1)
    for k ∈ 1:n
        Q[:, n + k] .*= sign(E.values[k])
    end
    #=
    G = sqrt(2) * Q'* ([Mr -Mi])
    display(norm(G[1:n, 1:n] - G[n+1:2n, n+1:2n]))
    display(norm(G[1:n, 1:n] + G[n+1:2n, n+1:2n]))
    =#
    return Q, abs.(E.values)
end
