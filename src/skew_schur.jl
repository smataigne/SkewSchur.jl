using LinearAlgebra, SkewLinearAlgebra

@views function even_odd_perm!(A::AbstractMatrix)
    n = size(A, 1)
    Base.permutecols!!(A, [1:2:n;2:2:n])
end

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

function hermitian_eigen(A::AbstractMatrix) 
    E = eigen(Hermitian(1im * A))
    Λ = Diagonal(complex.(0, -E.values))
    return E.vectors, Λ
end