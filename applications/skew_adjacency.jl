using LinearAlgebra, LaTeXStrings, Plots
include("..src/other_schur.jl")
include("..src/hamiltoschur.jl")
#=
M =
[0 0.9 0 0 0 0 0 0 0.1 0 0 0;
0 0 1 0 0 0 0 0 0 0 0 0;
0 0 0 1 0 0 0 0 0 0 0 0;
0 0 0 0 1 0 0 0 0 0 0 0;
0 0 0 0 0 0.9 0 0 0 0 0.1 0;
0 0 0 0 0 0 1 0 0 0 0 0;
0 0 0 0 0 0 0 1 0 0 0 0;
1 0 0 0 0 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0 0 1 0 0;
0 0 0.1 0 0 0 0 0 0 0 0.9 0;
0 0 0 0 0 0 0 0 0 0 0 1;
0 0 0 0 0 0 0.1 0 0.9 0 0 0]

# 12x12 matrix

A= -M + M'
Q, S = hamiltoschur!(A)
display(heatmap(abs.(Q[end:-1:1, 1:6]) + abs.(Q[end:-1:1, 7:12])))
display(S)
=#
#=
#Cycle of length 8
M = [0 1 0 0 0 0 0 0;
     0 0 1 0 0 0 0 0;
     0 0 0 1 0 0 0 0;
     0 0 0 0 1 0 0 0;
     0 0 0 0 0 1 0 0;
     0 0 0 0 0 0 1 0;
     0 0 0 0 0 0 0 1;
     1 0 0 0 0 0 0 0]
A = -M + M'
S = schur(A)
Z = S.Z
display(heatmap(abs.(Z[end:-1:1, :])))
display(S.values)

#Cycle of length 4
M = [0 1 0 0;
     0 0 1 0;
     0 0 0 1;
     1 0 0 0]
A = -M + M'
S = schur(A)
Z = S.Z
display(heatmap(abs.(Z[end:-1:1, :])))
display(S.values)
=#
#Cycles of length 4 plus 8 plus 16, without any perturbation
M = zeros(28, 28)
#Cycle of length 4
for k ∈ 1:4
    M[k, mod(k, 4) + 1] = 1
end
#Cycle of length 8
for k ∈ 5:12
    M[k, mod(k - 4, 8) + 5] = 1
end
#Cycle of length 16
for k ∈ 13:28
    M[k, mod(k - 12, 16) + 13] = 1
end

α = 0.9999;  β = 1 - α
#Connect inner cycle with medium cycle
M[2, 3] = α; M[2, 7] = β
M[4, 1] = α; M[4, 11] = β
M[5, 6] = α; M[5, 1] = β
M[9, 10] = α; M[9, 3] = β

#Connect medium cycle with outer cycle
M[13, 14] = α; M[13, 5] = β
M[6, 7] = α; M[6, 15] = β
M[17, 18] = α; M[17, 7] = β
M[8, 9] = α; M[8, 19] = β
M[21, 22] = α; M[21, 9] = β
M[10, 11] = α; M[10, 23] = β
M[25, 26] = α; M[25, 11] = β
M[12, 5] = α; M[12, 27] = β


A= -M + M'
E = randn(28, 28)
#display(heatmap(abs.(A[end:-1:1, :])))

Q, S = hamiltoschur!(A)
perm = sortperm(S, rev = true)
Q1 = Q[:, 1:14]
Q1 = Q1[:, perm]
Q2 = Q[:, 15:28]
Q2 = Q2[:, perm]

P = plot(framestyle=:box, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topright, size = (500,500))
plot!(1:28, Q1[:, 1].^2 + Q2[:, 1].^2, linestyle=:solid, color=:blue, label=L"q_{1}^2 + q_{n+1}^2", linewidth=3)
plot!(P, 1:28, Q1[:, 2].^2 + Q2[:, 2].^2, linestyle=:dash, color=:red, label=L"q_{2}^2 + q_{n+2}^2", linewidth=3)
plot!(P, 1:28, Q1[:, 3].^2 + Q2[:, 3].^2, linestyle=:dot, color=:green, label=L"q_{3}^2 + q_{n+3}^2", linewidth=3)
xlabel!(P, "Vertex indices")
α = round(1 - α, digits=2)
title!(P, L"\alpha = " * "$(α)")
savefig(P, joinpath(@__DIR__, "./figures/graph_clustering_alpha_$(α).pdf"))
#display(P)
display(heatmap(Q1 .^2 .+ Q2 .^2, ))
println("Done!")




