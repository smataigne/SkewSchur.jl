using LinearAlgebra, Plots

n = 1000
L = 5
dx = 2L / n
x = LinRange(-L, L - dx, n)

f(x::Real) = exp(-x^2) * cos(5 * pi * x / L)^2
g0(x::Real) = 2 / (π√(15)) * atan(√(3/5) * tan(π * x / 5))
g(x::Real) = g0(x) + 2 / √(15) * floor((x + L/2) / L)
c(x::Real) = 10 + 5/2 * cos(2π/5 * x)
∇g(x::Real) = 1 / c(x)
#Δg(x) = π * sin(π * x / 2) / (5 * (4 + cos(π * x / 2))^2)


P = plot()
s = zeros(n)
cst = zeros(n)
for t ∈ [0.0, 0.333, 0.667, 1]
    #Newton Raphson in s for the equation g(s) + t - g(x) = 0.
    @. s = x - t
    cst .= t .- g.(x)
    itermax = 20
    ds = ones(n)
    while norm(ds) > 1e-15 && itermax > 0
        itermax -= 1
        @. ds = (g(s) + cst) / ∇g(s)
        @. s -= ds
    end
    display(norm(ds))
    @. s = mod(s + L, 2L) - L 
    plot!(x, c.(s) ./ c.(x) .* f.(s), label = "t = $(round(t, digits=2))", titlefontfamily = "Computer Modern", titlefontsize = 15, linewidth = 3)
end
display(P)

