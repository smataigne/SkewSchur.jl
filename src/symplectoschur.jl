using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools
include("polar.jl")

"""
    signhalley!(X::AbstractMatrix{T}) where T

`X` is assumed orthogonal with no real eigenvalues. Computes the `1im * sign(1im * X)` of a matrix using Halley's method. The input matrix `X` is modified in place to contain the sign of `1im * sign(1im * X)`. The function returns the modified matrix `X`.

"""
@views function signhalley!(X::AbstractMatrix{T}) where T
    n = size(X, 1)
    ε =  max(10, sqrt(n)) * eps(T)
    X ./= norm(X)
    e = 1
    H = zeros(n, n)
    iter = 30
    temp = zeros(n, n)
    while e > ε && iter > 0
        #mul!(H, X, X, -1, 0)
        BLAS.syrk!('U', 'T', -1.0, X, 0.0, H)
        mul!(temp, X, H, 1, 0)
        X .*= 3
        X .+= temp
        H .*= 3
        H .+= I(n)
        X = X / H
        #X = X * (3I - X*X) * inv(I - 3X*X)
        e = norm(X + X') / (2 * sqrt(n))
        #display(e)
        iter -= 1
    end
    skewhermitian!(X)
    return X   
end

"""
    symplectoschur!(X::AbstractMatrix{T}; skew = true, polar = :qr) where T
Compute the Schur decomposition of a real matrix `X` of size `2n x 2n`.
The method returns an orthogonal matrix `Q` and two diagonal matrices `C` and `S` such that `X = Q * [C -S; S C] * Q'`.

"""
@views function symplectoschur!(X::AbstractMatrix{T}; skew = true, polar = :qr) where T
    n = size(X, 1) ÷ 2
    Q = zeros(2n, 2n)
    U1 = zeros(T, n, n)
    U2 = zeros(T, n, n)
    #use Q as temp to reduce allocations
    if skew
        Q .= X
        Q .-= X'
        Q .*= 0.5
        if polar == :qr 
            P = polarqr!(Q)
        elseif polar == :chol
            P = polarchol!(Q)
        elseif polar == :hybrid
            P = polarhybrid!(Q)

        elseif polar == :newton || polar == :newtonsafe
            if polar == :newtonsafe
                P = polarnewtonschulz!(Q; safe = true)
            else
                P = polarnewtonschulz!(Q)
            end
        else
            P = polarsvd!(Q)
        end
    else
        P = signhalley!(copy(X))
    end
    V = randn(T, 2n, n)
    temp = zeros(2n, n)
    mul!(temp, P, V, 1, 0)
    U = complex.(temp[:, 1:n], V)
    M = Matrix(qr(U).Q)
    #Use Q as temp
    temp = Q[:, 1:n]
    Mr = real.(M) .* sqrt(2)    #reduce allocations
    Mi = imag.(M) .* (-sqrt(2))
     #reduce allocations
    mul!(temp, X, Mr , 1, 0)
    mul!(U1, Mr', temp, 1, 0)
    mul!(U2, Mi', temp, 1, 0)
    H2 = complex.(U1, U2)
    E = schur!(H2)
    U1 .= real.(E.vectors)    #reduce allocations
    U2 .= imag.(E.vectors)
    mul!(Q[:, 1:n], Mr, U1, 1, 0)
    mul!(Q[:, 1:n], Mi, U2, 1, 1)
    mul!(Q[:, n+1:2n], Mr, U2, -1, 0)
    mul!(Q[:, n+1:2n], Mi, U1, 1, 1)
    C = Diagonal(real.(E.values))
    S = Diagonal(imag.(E.values))
    return Q, C, S
end

"""
    symplectoschur(X::AbstractMatrix{T}; skew = true, polar = :qr) where T
Compute the Schur decomposition of a real matrix `X` of size `2n x 2n`.
The method returns an orthogonal matrix `Q` and two diagonal matrices `C` and `S` such that `X = Q * [C -S; S C] * Q'`.

"""
symplectoschur(X::AbstractMatrix{T}; skew = true, polar = :qr) where T = symplectoschur!(copy(X); skew = skew, polar = polar)
