using LinearAlgebra, Plots, SkewLinearAlgebra
include("hamiltoschur.jl")

#=
Numerical simulation of the transport equation

du/dt + ∇ ⋅ (cu) = 0 with u ≥ 0 and x,y ∈ [-1,1].

By the change of variable u = w^2, we have

dw/dt + ⟨ c , ∇w ⟩ + 1/2 * ∇ ⋅ c  * w = 0,

with c(x, y) and u(x, y, t).
=#

#Parameters
BLAS.set_num_threads(12)
nx = 100
n = nx * nx
n2 = n ÷ 2
dx = 2 / (nx - 1)
A = zeros(n, n)
x = LinRange(-1, 1, nx)
cx = zeros(n)
cy = zeros(n)
Dx = zeros(n, n)
Dy = zeros(n, n)
ui = zeros(n)
qtwi = zeros(n)
temp = zeros(n)
wf = zeros(n)

#velocity function c(x,y)
function velocity(x::Number,y::Number)
    θ = atan(y, x)
    θ2 = atan(y / x)
    d = (abs(θ2) < π/4 ? hypot(1, tan(θ)) : hypot(1, 1/tan(θ)))
    r = hypot(x, y)
    v = (r) * (1 - r) 
    #α = (r / d - 0.5) * π / 2
    α = 0

    cx = (cos(α) * -sin(θ) - sin(α) * cos(θ)) * v
    cy = (sin(α) * sin(θ) + cos(α) *  cos(θ)) * v
    return cx, cy
end

for j ∈ 0:nx-1
    for i ∈ 1:nx
    cx[j * nx + i], cy[j * nx + i] = velocity(x[i], x[j+1])
    end
end

Cx = Diagonal(cx)
Cy = Diagonal(cy)

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

A = -0.5 *(Cx * Dx + Dx * Cx + Cy * Dy + Dy * Cy)
print("Matrix done\n")
V = Matrix(qr(randn(n, n)).Q)
print("QR done\n")
A = V' * A * V
skewhermitian!(A)
Q2, v = hamiltoschur!(A; polar =:qr)
print("Schur done\n")
Q = V * Q2
Σ = Diagonal(v)
#=
display(norm(A + A'))
Q, v = hamiltoschur!(A; polar =:qr)
Σ = Diagonal(v)
=#
display(norm(A * Q2 - Q2 * [zeros(n2, n2) -Σ; Σ zeros(n2,n2)]) / norm(A))



function initial_density(x::Number,y::Number)
    θ = atan(y, x)
    θ2 = atan(y / x)
    d = (abs(θ2) < π/4 ? hypot(1, tan(θ)) : hypot(1, 1/tan(θ)))
    r = hypot(x, y)
    κ = (r > 1 ? 0 : 1)
    ρ = (r / d) * (1-(r/ d)) * cos(θ)^2 * κ
    return ρ
end

for j ∈ 0:nx-1
    for i ∈ 1:nx
        ui[j * nx + i] = initial_density(x[i], x[j + 1])
    end
end

mul!(qtwi, Q', sqrt.(ui), 1, 0)

P = plot(framestyle=:none)
dt = 0.1
tf = 240 * dt
dt = 0.1
anim = @animate for t ∈ [0:dt:tf;tf:-dt:0]
    Σ = t * Diagonal(v)
    S = [cos(Σ) -sin(Σ); sin(Σ) cos(Σ)]
    mul!(temp, S, qtwi, 1, 0)
    mul!(wf, Q, temp, 1, 0)
    heatmap(P, reshape(wf.^2, nx, nx), colorbar=false, clims=(0, maximum(wf.^2)), xticks =false, yticks = false, aspect_ratio =:equal)
end
script_dir = @__DIR__
path = joinpath(script_dir, "./figures/transport2Dbis.gif")
gif(anim, path, fps = 24)
