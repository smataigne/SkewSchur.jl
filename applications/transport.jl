using LinearAlgebra, Plots, SkewLinearAlgebra, LaTeXStrings, BenchmarkTools
include("../src/hamiltoschur.jl")

#= Simulation of

uₜ + (cu)ₓ = 0 with u ≥ 0, x ∈ [-L, L],

by the change of variable w = √u and

wₜ + c * wₓ + 0.5 * cₓ * w = 0

<=> wₜ + 0.5 * (cw)ₓ + 0.5 * c * wₓ = 0,

with c(x) and w(x, t)
=#

n = 500
n2 = n ÷ 2
L = 5

c(x::Number) = 2L + L/2 * cos(2π/L * x)
f(x::Number) = exp(-x^2) * cos(π * x)^2
g0(x::Real) = 2 / (π√(15)) * atan(√(3/5) * tan(π * x / 5))
g(x::Real) = g0(x) + 2 / √(15) * floor((x + L/2) / L)
∇g(x::Real) = 1 / c(x)

dx = 2L / n
A = zeros(n , n)
x = LinRange(-L, L - dx, n)
s = zeros(n)
cst = zeros(n)
ui = zeros(n)
@. ui = f.(x)
wi = zeros(n)
@. wi = exp(-(x^2)/2) * cos(π * x)
qtwi = zeros(n)
temp = zeros(n)
wf = zeros(n)
D = Diagonal(c.(x))

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

#Discretize wₜ + 0.5 * (cw)ₓ + 0.5 * c * wₓ = 0.
A = -0.5 * (D * A + A * D) 
skewhermitian!(A)
E = randn(n, n)
E = E - E'
E .*= norm(A) * eps(Float64) / norm(E)
A .+= E
B = copy(A)
BLAS.set_num_threads(4)
Q, v = hamiltoschur!(copy(A); polar =:qr)
#tt = minimum(@elapsed hamiltoschur!(copy(A); polar =:qr) for i in 1:5)
#display(tt)
#display(norm(A * Q[:, 1:2n2] - Q[:, 1:2n2] * [zeros(n2, n2) -Σ; Σ zeros(n2,n2)]) / norm(A))




mul!(qtwi, Q', wi, 1, 0)
Σ = Diagonal(v)
S = [cos(Σ) -sin(Σ); sin(Σ) cos(Σ)]
mul!(temp, S, qtwi, 1, 0)
mul!(wf, Q, temp, 1, 0)

P = plot(framestyle=:box, fontfamily = "Computer Modern", tickfontsize = 12)
P1 = plot()
P2 = plot()
P3 = plot()
P4 = plot()
PS = [P1, P2, P3, P4]
colors = [:black, :lightskyblue1 , :magenta]
linestyles = [:solid, :dashdot, :solid]
w_euler = zeros(n)
tf = 3.0984
dt = tf/3 /100
CN = (I - dt/2 * B) \ (I + dt/2 * B) #Crank-Nicolson matrix
for (i, t) ∈ enumerate([0, tf/3, 2tf/3, tf])
    global PS, temp, wf, qtwi, B, w_euler, dt
    #Spectral integration
    Σ = t * Diagonal(v)
    S = [cos(Σ) -sin(Σ); sin(Σ) cos(Σ)]
    mul!(temp, S, qtwi, 1, 0)
    mul!(wf, Q, temp, 1, 0)
    
    #Crank-Nicolson integration
    w_euler .= copy(wi)
    for _ in 1:round(Int, t / dt)
        w_euler = CN * w_euler
    end
    #Analytic solution with Newton-Raphson in s for the equation g(s) + t - g(x) = 0.
    @. s = x - t
    cst .= t .- g.(x)
    itermax = 20
    ds = ones(n)
    while norm(ds) > 1e-15 && itermax > 0
        itermax -= 1
        @. ds = (g(s) + cst) / ∇g(s)
        @. s -= ds
    end
    @. s = mod(s + L, 2L) - L 
    #Plots
    plot!(PS[i], x, c.(s) ./ c.(x) .* f.(s), label = false, linewidth = 6, color=colors[1])
    plot!(PS[i], x, wf .^2, label = false, titlefontfamily = "Computer Modern", titlefontsize = 15, linewidth = 3, color=colors[2], linestyle=linestyles[2])
    plot!(PS[i], x, w_euler .^2, label = false, linestyle=linestyles[3], linewidth = 2, color=colors[3])
    ylims!(PS[i], -0.1, 1.1)
    title!(PS[i], L"t = "*"$(round(t, digits=2))")
end

labels = ["Integration with characteristics", "Integration with RSD", "Integration with Crank-Nicolson"]
n = length(labels)
P_legend = plot(
    [NaN], [NaN],
    label = labels[1],
    color = colors[1],
    linestyle = linestyles[1],
    linewidth = 3,
    legend = :inside,
    legendcolumns = 4,
    framestyle = :none,
    axis = false,
    grid = false,
    margin = 0Plots.mm
)
for i in 2:n
    plot!(P_legend, [NaN], [NaN], label = labels[i], color = colors[i], linestyle = linestyles[i], linewidth = 3)
end
# Assembler : 3 pie charts + 1 plot légende en bas
P = plot(
    PS..., P_legend,
    layout = @layout([a b c d; e{0.1h}]),   # 5% instead of 10% for legend row
    size = (1500, 450),                     # wider and taller overall
    ticksfontfamily    = "Computer Modern",
    fontfamily         = "Computer Modern",
    legendfontfamily   = "Computer Modern",
    titlefontfamily    = "Computer Modern",
    framestyle=:box,
    legendfontsize  = 15,
    labelfontsize   = 10,
    tickfontsize    = 10,
    titlefontsize   = 15,
    margin = 5Plots.mm
)
savefig(P, joinpath(@__DIR__, "../figures/transport_times.pdf"))
display(P)


#Animation
#=
P = plot(framestyle=:box, fontfamily = "Computer Modern", tickfontsize = 12)
#plot!(x, ui, label = "Initial condition")
anim = @animate for t ∈ 0:0.01:1
    Σ = t * Diagonal(v)
    S = [cos(Σ) -sin(Σ); sin(Σ) cos(Σ)]
    mul!(temp, S, qtwi, 1, 0)
    mul!(wf, Q, temp, 1, 0)
    plot(P, x, wf .^2, label = false)
    ylims!(-0.5, 2)
end
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/transport.gif")
gif(anim, path, fps = 24)
=#

