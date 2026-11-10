using LinearAlgebra, SkewLinearAlgebra, Plots
include("../src/hamiltoschur.jl")

"""

    2D Wave Equation
    
uₜₜ - c² (uₓₓ + uᵧᵧ) = 0 with (x, y) ∈ [-1, 1]²,

"""

BLAS.set_num_threads(12)
nx = 70
n = nx * nx
n2 = n ÷ 2
dx = 2 / (nx - 1)
ui = zeros(n)
wx = zeros(n)
wy = zeros(n)
x = LinRange(-1, 1, nx)
c = 1
 
for j ∈ 0:nx-1
    for i ∈ 1:nx
        α = 20
        ui[j * nx + i] = exp(-α* ((x[i] - 0.5)^2 + x[j+1]^2)) + exp(-α * ((x[i] + 0.5)^2 + x[j+1]^2))
        wx[j * nx + i] = c * (- 2α * (x[i] -0.5) * exp(-α * ((x[i] - 0.5)^2 + x[j+1]^2)) - 2α * (x[i] +0.5) * exp(-α * ((x[i] + 0.5)^2 + x[j+1]^2)))
        wy[j * nx + i] = c * (- 2α * x[j+1] * exp(-α * ((x[i] - 0.5)^2 + x[j+1]^2)) - 2α * x[j+1] * exp(-α * ((x[i] + 0.5)^2 + x[j+1]^2)))
    end
end
vw = [zeros(n); wx; wy]
#=
Dx = zeros(n, n)
Dy = zeros(n, n)



for j ∈ 0:nx-1
    for i ∈ 1:nx
        jj = j * nx + i
        d = 60dx
        Dx[jj, j * nx +  mod1(i + 1, nx)] = 45  / d
        Dx[jj, j * nx +  mod1(i + 2, nx)] = -9  / d
        Dx[jj, j * nx +  mod1(i + 3, nx)] = 1  / d
        Dx[jj, j * nx +  mod1(i - 1, nx)] = -45  / d
        Dx[jj, j * nx +  mod1(i - 2, nx)] = 9  / d
        Dx[jj, j * nx +  mod1(i - 3, nx)] = -1  / d
        Dy[jj, mod(j + 1, nx) * nx +  i] = 45 / d
        Dy[jj, mod(j + 2, nx) * nx +  i] = -9 / d
        Dy[jj, mod(j + 3, nx) * nx +  i] = 1 / d
        Dy[jj, mod(j - 1, nx) * nx +  i] = -45 / d
        Dy[jj, mod(j - 2, nx) * nx +  i] = 9 / d
        Dy[jj, mod(j - 3, nx) * nx +  i] = -1 / d
    end
end
dt = 0.005
E = svd(c .* [Dx; Dy])
C = cos(Diagonal(E.S) .* dt)
S = sin(Diagonal(E.S) .* dt)

Exp = I(3n) + [(E.V * (C - I(n)) * E.Vt)  (-E.V * S * E.U'); (E.U * S * E.Vt)  E.U * (C - I(n)) * E.U']
#A = c * [zeros(n, n) Dx Dy; Dx zeros(n, 2n) ; Dy zeros(n, 2n)]
#CN = (I - dt/2 * A) \ (I + dt/2 * A)
=#
P = plot(framestyle=:none)


u = copy(ui)
oldv = copy(vw[1:n])

anim = @animate for t ∈ 0:dt:10
    oldv .= vw[1:n]
    vw .= Exp * vw
    u .+= dt/2 * (vw[1:n] .+ oldv)
    heatmap(P, reshape(u, nx, nx), colorbar=false, clims=(-1, 1), xticks =false, yticks = false, aspect_ratio =:equal, color=:seismic)
end
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/wave2Dbis.gif")
gif(anim, path, fps = 24)

#=
for t ∈ 0:dt:7.5
    oldv .= vw[1:n]
    vw .= Exp * vw
    u .+= dt/2 * (vw[1:n] .+ oldv)
end

l = maximum(abs.(u))
heatmap!(P, reshape(u, nx, nx), clims=(-l, l), colorbar=false, xticks =false, yticks = false, aspect_ratio =:equal, color=:seismic, interpolate =true)
display(P)
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/wave2Dbis.pdf")
savefig(P, path)
=#

