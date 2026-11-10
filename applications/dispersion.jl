using LinearAlgebra, Plots, SkewLinearAlgebra, LaTeXStrings, BenchmarkTools
include("../src/hamiltoschur.jl")

#= Simulation of

uₜ + uₓₓₓ = 0 with x ∈ [-L, L],

=#

n = 500
L = 5

f(x::Number) = exp(-x^2) * cos(6π * x)

dx = 2L / n
A = zeros(n , n)
x = LinRange(-L, L - dx, n)
ui = zeros(n)
@. ui = f.(x)
qtui = zeros(n)
temp = zeros(n)
uf = zeros(n)

for i in 1:n
    #Sixth order FD
    d = 60dx
    A[i, mod1(i + 1, n)]  = 45 / d
    A[i, mod1(i + 2, n)]  =  -9 / d
    A[i, mod1(i + 3, n)]  = 1 / d
    A[i, mod1(i - 1, n)] = -45 / d
    A[i, mod1(i - 2, n)] = 9 / d
    A[i, mod1(i - 3, n)] = -1 / d
end

A = 0.005 * A * A * A
skewhermitian!(A)
E = randn(n, n)
E = E - E'
E .*= norm(A) * eps(Float64) / norm(E)
A .+= E
B = copy(A)
BLAS.set_num_threads(4)
Q, Σ = hamiltoschur!(copy(A); polar =:qr)
mul!(qtui, Q', ui, 1, 0)
P = plot(framestyle=:box, fontfamily = "Computer Modern", tickfontsize = 12)
dt = 0.02
anim = @animate for t ∈ 0:dt:10
    Σt = t * Σ
    S  = [cos(Σt) -sin(Σt); sin(Σt) cos(Σt)]
    mul!(temp, S, qtui, 1, 0)
    mul!(uf, Q, temp, 1, 0)
    plot(P, x, uf, label = false)
    ylims!(-2, 2)
end
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/dispersion.gif")
gif(anim, path, fps = 24)