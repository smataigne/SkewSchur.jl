using LinearAlgebra, SkewLinearAlgebra

function coef(l::Number)
    d = cbrt(4 * (1 - l * l)/ l^4)
    c = sqrt(1 + d)
    return c + 0.5 * sqrt(8 - 4 * d + 8 * (2 - l * l) / (l * l * c))
end

@views function polarsvd!(X::AbstractMatrix{T}) where T
    n = size(X, 1)
    P = zeros(n, n)
    S = svd!(X)
    mul!(P, S.U, S.Vt, 1, 0)
    skewhermitian!(P)
    return P
end

#QDWH algroithm for polar decomposition
#Nakatsukasa, Bai, Gygi, "Optimizing Halley's iteration for computing the matrix polar decomposition", SIAM J. Matrix Anal. Appl., 2010.
#Please Copilot Write Documentation for all functions in this file

"""
    polarqr!(X::AbstractMatrix{T}) where T
Compute the polar decomposition of a skew-symmetric matrix `X` using the QR-based Halley method of Nakatsukasa, Bai, Gygi, "Optimizing Halley's iteration for computing the matrix polar decomposition", SIAM J. Matrix Anal. Appl., 2010.

Returns the orthogonal factor `P` of the polar decomposition `X = P * H`, where `H` is a positive semi-definite Hermitian matrix.

The polar factor `P` is skew-symmetric if the dimensions of `X` are even.
"""
@views function polarqr!(X::AbstractMatrix{T}) where T
    n = size(X, 1)
    ε = 10 * sqrt(n) * eps(T)
    α = norm(X)
    β = eps(T)
    try
        #β = 1 / (opnorm(inv(X), 1) * sqrt(n))
        F = LinearAlgebra.LAPACK.getrf!(copy(X))      # LU factors
        anorm = opnorm(X, 1)
        rcond = LinearAlgebra.LAPACK.gecon!('1', F[1], anorm)
        invnorm1 = 1/(rcond*anorm) 
        β = 1 / (invnorm1 * sqrt(n)) 
    catch
        β = eps(T)
    end
    X ./= α
    l = β / α
    temp = zeros(2n, n)
    Q = zeros(2n, n)
    H = zeros(n, n)
    e = 1
    while e > ε
        a = coef(l)
        b = (a - 1)^2 / 4
        c = a + b - 1
        temp[1:n, 1:n] .= sqrt(c) * X
        temp[n+1:2n, 1:n] .= I(n)
        Q .= Matrix(qr(temp).Q)
        mul!(X, Q[1:n, 1:n], Q[n+1:2n, 1:n]', (a - b/c)/sqrt(c), b / c)
        (iseven(n) ? skewhermitian!(X) : nothing)
        l = l * (a + b *l *l)/(1 + c * l * l)
        mul!(H, X', X, 1, 0)
        e = norm(H - I(n))
    end 
    return X
end

"""
    polarchol!(X::AbstractMatrix{T}) where T
Compute the polar decomposition of a skew-symmetric matrix `X` using the Cholesky-based Halley method of Nakatsukasa, Bai, Gygi, "Optimizing Halley's iteration for computing the matrix polar decomposition", SIAM J. Matrix Anal. Appl., 2010.

Returns the orthogonal factor `P` of the polar decomposition `X = P * H`, where `H` is a positive semi-definite Hermitian matrix.

The polar factor `P` is skew-symmetric if the dimensions of `X` are even.

The method requires that `X`is well-conditioned.
""" 
@views function polarchol!(X::AbstractMatrix{T}) where T
    n = size(X, 1)
    ε =  10 * sqrt(n) * eps(T)
    α = norm(X)
    β = eps(T)
    try
        #β = 1 / (opnorm(inv(X), 1) * sqrt(n))
        F = LinearAlgebra.LAPACK.getrf!(copy(X))      # LU factors
        anorm = opnorm(X, 1)
        rcond = LinearAlgebra.LAPACK.gecon!('1', F[1], anorm)
        invnorm1 = 1/(rcond*anorm) 
        β = 1 / (invnorm1 * sqrt(n)) 
    catch
        β = eps(T)
    end
    X ./= α
    l = β / α
    temp = zeros(n, n)
    H = zeros(n, n)
    mul!(H, X', X, 1, 0)
    e = 1
    while e > ε
        a = coef(l)
        b = (a - 1)^2 / 4
        c = a + b - 1
        H .*= c
        H .+= I(n)
        temp = X / Symmetric(H)
        X .*= (b/c)
        X .+= (a - b/c) .* temp
        skewhermitian!(X)
        l = l * (a + b *l *l)/(1 + c * l * l)
        BLAS.syrk!('U', 'T', 1.0, X, 0.0, H)
        #mul!(H, X', X, 1, 0)
        e = norm(Symmetric(H) - I(n))
    end 
    return X
end

"""
    polarhybrid!(X::AbstractMatrix{T}) where T
Compute the polar decomposition of a skew-symmetric matrix `X` using a hybrid method that combines the QR-based and Cholesky-based Halley methods of Nakatsukasa, Bai, Gygi, "Optimizing Halley's iteration for computing the matrix polar decomposition", SIAM J. Matrix Anal. Appl., 2010.

Returns the orthogonal factor `P` of the polar decomposition `X = P * H`, where `H` is a positive semi-definite Hermitian matrix.

The polar factor `P` is skew-symmetric if the dimensions of `X` are even.

The method is designed to be more robust than the Cholesky-based method, and can handle matrices that are not well-conditioned.
"""
@views function polarhybrid!(X::AbstractMatrix{T}) where T
    n = size(X, 1)
    ε =  10 * sqrt(n) * eps(T)
    α = norm(X)
    β = eps(T)
    try
        #β = 1 / (opnorm(inv(X), 1) * sqrt(n))
        F = LinearAlgebra.LAPACK.getrf!(copy(X))      # LU factors
        anorm = opnorm(X, 1)
        rcond = LinearAlgebra.LAPACK.gecon!('1', F[1], anorm)
        invnorm1 = 1/(rcond*anorm) 
        β = 1 / (invnorm1 * sqrt(n)) 
    catch
        β = eps(T)
    end
    X ./= α
    l = β / α
    temp = zeros(2n, n)
    Q = zeros(2n, n)
    H = zeros(n, n)
    e = 1
    a = coef(l)
    b = (a - 1)^2 / 4
    c = a + b - 1
    while c > 1000
        temp[1:n, 1:n] .= sqrt(c) * X
        temp[n+1:2n, 1:n] .= I(n)
        Q .= Matrix(qr(temp).Q)
        mul!(X, Q[1:n, 1:n], Q[n+1:2n, 1:n]', (a - b/c)/sqrt(c), b / c)
        skewhermitian!(X)
        l = l * (a + b *l *l)/(1 + c * l * l)
        BLAS.syrk!('U', 'T', 1.0, X, 0.0, H)
        #mul!(H, X', X, 1, 0)
        e = norm(H - I(n))
        a = coef(l)
        b = (a - 1)^2 / 4
        c = a + b - 1
    end
    temp = zeros(n, n)
    BLAS.syrk!('U', 'T', 1.0, X, 0.0, H)
    #mul!(H, X', X, 1, 0)
    while e > ε
        H .*= c
        H .+= I(n)
        temp = X / Symmetric(H)
        X .*= (b/c)
        X .+= (a - b/c) .* temp
        skewhermitian!(X)
        l = l * (a + b *l *l)/(1 + c * l * l)
        BLAS.syrk!('U', 'T', 1.0, X, 0.0, H)
        #mul!(H, X', X, 1, 0)
        e = norm(Symmetric(H) - I(n))
        a = coef(l)
        b = (a - 1)^2 / 4
        c = a + b - 1
    end 
    return X
end

#Halley's iteration for polar decomposition
"""
    polarhalley!(X::AbstractMatrix{T}) where T

Compute the polar decomposition of a skew-symmetric matrix `X` using Halley's iteration.

Returns the orthogonal factor `P` of the polar decomposition `X = P * H`, where `H` is a positive semi-definite Hermitian matrix.

The polar factor `P` is skew-symmetric if the dimensions of `X` are even.
"""
@views function polarhalley!(X::AbstractMatrix{T}) where T
    n = size(X, 1)
    ε = 10 * sqrt(n) * eps(T)
    #X ./= norm(X)
    e = 1
    H = zeros(n, n)
    BLAS.syrk!('U', 'T', 1.0, X, 0.0, H) #H = X' * X
    temp = zeros(n, n)
    while e > ε
        H .*= 3
        H .+= I(n)
        temp = X / Symmetric(H)
        X .*= (1/3)
        X .+= (3 - 1/3) .* temp
        skewhermitian!(X)
        BLAS.syrk!('U', 'T', 1.0, X, 0.0, H) #H = X' * X
        e = norm(H - I(n))
    end
    return X   
end

"""
    polarnewtonschulz!(X::AbstractMatrix{T}) where T

Compute the polar decomposition of a skew-symmetric matrix `X` using the Newton-Schulz iteration.

Returns the orthogonal factor `P` of the polar decomposition `X = P * H`, where `H` is a positive semi-definite Hermitian matrix.

The polar factor `P` is skew-symmetric if the dimensions of `X` are even.
"""
@views function polarnewtonschulz!(X::AbstractMatrix{T}; pre_step = false) where T
    n = size(X, 1)
    ε = 10 * sqrt(n) * eps(T)
    H = zeros(n, n)
    if pre_step
        #Initial step of QDWH algorithm to get a good initial point for the Newton-Schulz iteration.
        α = norm(X)
        β = eps(T)
        try
            F = LinearAlgebra.LAPACK.getrf!(copy(X))      # LU factors
            anorm = opnorm(X, 1)
            rcond = LinearAlgebra.LAPACK.gecon!('1', F[1], anorm)
            invnorm1 = 1/(rcond*anorm) 
            β = 1 / (invnorm1 * sqrt(n)) 
        catch
            β = eps(T)
        end
        X ./= α
        l = β / α
        temp = zeros(2n, n)
        Q = zeros(2n, n)
        
        a = coef(l)
        b = (a - 1)^2 / 4
        c = a + b - 1
        temp[1:n, 1:n] .= sqrt(c) * X
        temp[n+1:2n, 1:n] .= I(n)
        Q .= Matrix(qr(temp).Q)
        mul!(X, Q[1:n, 1:n], Q[n+1:2n, 1:n]', (a - b/c)/sqrt(c), b / c)
        skewhermitian!(X)
        l = l * (a + b *l *l)/(1 + c * l * l)
    end
    BLAS.syrk!('U', 'T', 1.0, X, 0.0, H) #H = X' * X
    e = norm(H - I(n))
    #Newton-Schulz iteration
    temp = zeros(n, n)
    iter = 0; itermax = 5 * sqrt(n)
    while e > ε && iter < itermax
        iter += 1
        mul!(temp, X, Symmetric(H), 1/2, 0)
        X .*= 3/2        
        X .-= temp
        skewhermitian!(X)
        BLAS.syrk!('U', 'T', 1.0, X, 0.0, H) #H = X' * X
        e = norm(H - I(n))
    end
    if iter == itermax
        @warn "Newton-Schulz iteration did not converge in $itermax iterations"
    end
    return X
end

