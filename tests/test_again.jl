using LinearAlgebra

n = 2
A = randn(2n, 2n)
A = A - A'  # Make A skew-symmetric

function work(A::AbstractMatrix)
    n = size(A, 1) ÷ 2
    for i = 2:n
        for j = 1:1
            M = [A[i, j] A[i, j+n]; A[i+n, j] A[i+n, j+n]]
            SVD = svd(M)
            t = sqrt(2)/2
            U = SVD.U * [t t; t -t]
            V = SVD.V * [t t; t -t]
            A[[i, i+n], :] = U' * A[[i, i+n], :]
            A[:, [j, j+n]] = A[:, [j, j+n]] * V
            A[:,[i, i+n]] = A[:, [i, i+n]] * U
            A[[j, j+n], :] = V' * A[[j, j+n], :]
        end
        display(A)
    end
    return A
end
A = work(A)
T = Diagonal([-1;1;1;-1])
display(T' * A * T)