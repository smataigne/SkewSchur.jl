using LinearAlgebra, SkewLinearAlgebra, Plots
include("../src/hamiltoschur.jl")


function c(x::Number)
    return 5 * (2 + cos(π * x))
end

n = 500
n2 = n ÷ 2
L = 5
dx = 2L / n
D = zeros(n , n)
x = LinRange(-L, L - dx, n)
ui = zeros(n)
tempu = zeros(n)
vi = zeros(n)
wi = zeros(n)
qtVi = zeros(2n)
temp = zeros(2n)
Vf = zeros(2n)
C = Diagonal(c.(x))

@. ui = exp(-x^2) / 2 * cos(π * x)^2
@. wi = c(x) * (-exp(-x^2) * 2x / 2 * cos(π * x)^2 - exp(-x^2) * cos(π * x) * sin(π * x) * π)
Vi = [vi; wi]

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

A = [zeros(n, n) D * C;C * D zeros(n, n)]
display(norm(A + A'))
V = Matrix(qr(randn(2n, 2n)).Q)
A = V' * A * V
skewhermitian!(A)
Q2, v = hamiltoschur!(A; polar =:qr)
Q = V * Q2
Σ = Diagonal(v)
display(norm(A * Q2 - Q2 * [zeros(n, n) -Σ; Σ zeros(n,n)]) / norm(A))
mul!(qtVi, Q', Vi, 1, 0)

P = plot(framestyle=:box, fontfamily = "Computer Modern", tickfontsize = 12)
dt = 0.02
#=
anim = @animate for t ∈ 0:dt:4
    #2 points Gauss Quadrature rule for u_t = v:  error ∈ O(dt^5).
    Σ = (t + dt * (1/2 - 1/(2√3))) * Diagonal(v)
    S = [cos(Σ) -sin(Σ); sin(Σ) cos(Σ)]
    mul!(temp, S, qtVi, 1, 0)
    mul!(Vf, Q, temp, 1, 0)
    tempu .= Vf[1:n]
    Σ = (t + dt * (1/2 + 1/(2√3))) * Diagonal(v)
    S = [cos(Σ) -sin(Σ); sin(Σ) cos(Σ)]
    mul!(temp, S, qtVi, 1, 0)
    mul!(Vf, Q, temp, 1, 0)
    @. ui += dt * (Vf[1:n] + tempu) / 2
    plot(P, x, ui, label = false)
    ylims!(-2, 2)
end
=#

anim = @animate for t ∈ 0:dt:4
    Σ = Diagonal(v)
    F = inv(Σ)
    S = [F*sin(Σ* t) F*(cos(Σ * t) - I);-F*(cos(Σ * t) -I) F*sin(Σ * t)]
    mul!(temp, S, qtVi, 1, 0)
    mul!(Vf, Q, temp, 1, 0)
    plot(P, x, ui .+ Vf[1:n], label = false)
    ylims!(-2, 2)
end
script_dir = @__DIR__
path = joinpath(script_dir, "./figures/wave.gif")
gif(anim, path, fps = 24)


