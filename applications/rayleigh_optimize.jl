using LinearAlgebra, Plots

n = 50
A = rand(n, n)
A = A * A'
sym(A::AbstractMatrix) = (A + A') / 2
skew(A::AbstractMatrix) = (A - A') / 2

f(A, X, N) = tr(X' * A * X * N)
@views function rayleigh_optimize(A::AbstractMatrix, X₀::AbstractMatrix, Retr::Function, f::Function, r::Real = 0.001)
    X = X₀
    p = size(X, 2)
    N = Diagonal(Vector(p:-1:1))
    itermax = 100
    α = 1
    τ = 0.5
    res = zeros(itermax)
    for i ∈ 1:itermax
        Y = 2 * A * X * N
        grad = Y - X * sym(X' * Y)
        #Armijo Backtracking line search
        fv = f(A, X, N)
        ng = tr(grad' * grad)
        while f(A, Retr(X, α * grad), N) < fv + r * α * ng
            α *= τ
        end
        X = Retr(X, α * grad)
        α /= τ
        res[i] = f(A, X, N)
    end
    return res
end

qr_retr(X, Y) = Matrix(qr(X + Y).Q)
exp_retr(X, Y) = X * exp(skew(X'Y))
X₀ = Matrix(qr(randn(n, n)).Q)
if det(X₀) < 0
    X₀[:, 1] = -X₀[:, 1]
end
res_exp = rayleigh_optimize(A, copy(X₀), exp_retr, f, 0.9)
res_qr = rayleigh_optimize(A, copy(X₀), qr_retr, f, 0.9)

res_exp2 = rayleigh_optimize(A, copy(X₀), exp_retr, f, 0.0001)
res_qr2 = rayleigh_optimize(A, copy(X₀), qr_retr, f, 0.0001)
P = plot(1:length(res_exp), res_exp, label = "Exp Retr", yscale = :log10, framestyle=:box, fontfamily = "Computer Modern", tickfontsize = 12)
plot!(1:length(res_qr), res_qr, label = "QR Retr")
plot!(1:length(res_exp2), res_exp2, label = "Exp Retr (r=0.0001)")
plot!(1:length(res_qr2), res_qr2, label = "QR Retr (r=0.0001)")
display(P)
