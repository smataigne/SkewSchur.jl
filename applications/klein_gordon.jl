using LinearAlgebra, SkewLinearAlgebra, Plots


function c(x::Number)
    return 5 * (2 + 0*cos(π * x))
end
m = 2
n = 500
n2 = n ÷ 2
L = 5
dx = 2L / n
D = zeros(n , n)
x = LinRange(-L, L - dx, n)
ui = zeros(n)
vi = zeros(n)
wi = zeros(n)
qtVi = zeros(3n)
temp = zeros(3n)
Vf = zeros(3n)
C = Diagonal(c.(x))

@. ui =  m * exp(-x^2) / 2
@. wi = - c(x) * exp(-x^2) * 2x / 2
Vi = [ui; vi; wi]

for i in 1:n
    #Sixth order FD
    d = 60dx
    D[i, mod1(i + 1, n)]  = 45 / d
    D[i, mod1(i + 2, n)]  =  -9 / d
    D[i, mod1(i + 3, n)]  = 1 / d
    D[i, mod1(i - 1, n)] = -45 / d
    D[i, mod1(i - 2, n)] = 9 / d
    D[i, mod1(i - 3, n)] = -1 / d
end

A = [zeros(n, n) m * I(n) zeros(n, n);-m * I(n) zeros(n, n)  D * C;zeros(n, n) C * D zeros(n, n)]

display(norm(A + A'))
V = Matrix(qr(randn(3n, 3n)).Q)
A = V' * A * V
skewhermitian!(A)
Q2, v = hamiltoschur!(A; polar =:qr)
Q = V * Q2
Σ = Diagonal(v)
mul!(qtVi, Q', Vi, 1, 0)

P = plot(framestyle=:box, fontfamily = "Computer Modern", tickfontsize = 12)
dt = 0.02
anim = @animate for t ∈ 0:dt:4
    Σ = t * Diagonal(v)
    S = [cos(Σ) -sin(Σ); sin(Σ) cos(Σ)]
    mul!(temp, S, qtVi, 1, 0)
    mul!(Vf, Q, temp, 1, 0)
    plot(P, x, Vf[1:n] ./ m, label = false)
    ylims!(-2, 2)
end
script_dir = @__DIR__
path = joinpath(script_dir, "./figures/klein_gordon.gif")
gif(anim, path, fps = 24)
