using LinearAlgebra, Plots, LaTeXStrings
include("hamiltoschur.jl")

M = [0. 3 0 3 0 0;
    0 0 3 2 0 0;
    0 0 0 0 4 2;
    0 0 0 0 2 0;
    0 0 0 0 0 3;
    0 0 0 0 0 0]
A = M - M'

E = randn(6, 6)
E = (E - E') * 1e-16
A = A + E
#display(heatmap(abs.(A[end:-1:1, :])))

Q, S = hamiltoschur!(A)
Q1 = Q[:, 1:3]
Q2 = Q[:, 4:6]

P = plot(framestyle=:box, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topright, size = (500,500))
plot!(1:6, Q1[:, 1].^2 + Q2[:, 1].^2, linestyle=:solid, color=:blue, label=L"q_{1}^2 + q_{n+1}^2", linewidth=3)
plot!(P, 1:6, Q1[:, 2].^2 + Q2[:, 2].^2, linestyle=:dash, color=:red, label=L"q_{2}^2 + q_{n+2}^2", linewidth=3)
plot!(P, 1:6, Q1[:, 3].^2 + Q2[:, 3].^2, linestyle=:dot, color=:green, label=L"q_{3}^2 + q_{n+3}^2", linewidth=3)
xlabel!(P, "Vertex indices")
display(P)
println("Done!")