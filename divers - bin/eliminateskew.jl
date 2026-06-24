using LinearAlgebra

function givens(a::Number, b::Number)
    r = hypot(a, b)
    c = a / r
    s = -b / r
    return c, s
end

#Function that eliminates element A[i, j] using Givens rotations, Moreover, it is a similarity transformation, so it preserves the eigenvalues of A.
@views function apply_givens!(A::AbstractMatrix{T}, i1::Int, i2::Int, j::Int) where T
    c, s = givens(A[i1, j], A[i2, j])
    G = [c -s; s c]
    A[[i1, i2], :] = G * A[[i1, i2], :]
    A[:, [i1, i2]] = A[:, [i1, i2]] * G'
end

A = randn(8, 8)
A = (A - A') / 2  # Make A skew-symmetric
v = randn(8)
v ./= norm(v)
y = A * A * v
y ./= norm(y)
display(A *(I - y * y') * v)
#=
#Eliminate upper-left block
apply_givens!(A, 6, 2, 1)
apply_givens!(A, 7, 3, 1)
apply_givens!(A, 8, 4, 1)
apply_givens!(A, 8, 5, 1)
apply_givens!(A, 8, 6, 1)
apply_givens!(A, 4, 3, 2)
apply_givens!(A, 5, 4, 2)
apply_givens!(A, 6, 5, 2)
apply_givens!(A, 5, 4, 3)
apply_givens!(A, 8, 7, 1)
apply_givens!(A, 7, 6, 2)
apply_givens!(A, 6, 5, 3)
#Eliminate lower right block
A = A[[5:8;1:4], [5:8;1:4]]
A = A[[1:4;8:-1:5],[1:4;8:-1:5]]
E  = schur(A[1:4, 1:4])
V = [E.Z zeros(4,4); zeros(4,4) E.Z]
#SVD = svd(A[5:8,1:4])
#V = [SVD.V zeros(4,4); zeros(4,4) SVD.U]
A = V' * A * V
kk = Array(1:8)
#invpermute!(kk, [1:2:8; 2:2:8])
A = A[kk, kk]
display(round.(A; digits=2))
=#


