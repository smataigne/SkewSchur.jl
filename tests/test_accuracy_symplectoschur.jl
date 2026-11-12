using LinearAlgebra, Plots, LaTeXStrings
include("../src/symplectoschur.jl")
include("../src/other_schur.jl")

BLAS.set_num_threads(64)
N = floor.(Int, 2 .^ LinRange(log2(32), log2(8192), 5)) .÷ 2
accuracy_schur = zeros(length(N))
#accuracy_symplectoschur_sign = zeros(length(N))
accuracy_symplectoschur_skew = zeros(length(N))
accuracy_randdiag = zeros(length(N))
Ntrials = 1
for (i, n) ∈ enumerate(N)
    for _ ∈ 1:Ntrials
        X = Matrix(qr(randn(2n, 2n)).Q)
        if det(X) < 0
            X[:,1] .= -X[:,1]
        end
        SchurX = schur(X)
        accuracy_schur[i] += log2(norm(X * SchurX.Z - SchurX.Z * SchurX.T) / norm(X)) / Ntrials 
        Q, C, S = symplectoschur!(copy(X); skew = true, polar = :hybrid)
        accuracy_symplectoschur_skew[i] += log2(norm(X * Q - Q * [C -S; S C]) / norm(X)) / Ntrials
        V, D = randdiag(copy(X))
        accuracy_randdiag[i] += log2(norm(X * V - V * D) / norm(X)) / Ntrials
    end 
end

accuracy_schur .= 2 .^ accuracy_schur
accuracy_symplectoschur_skew .= 2 .^ accuracy_symplectoschur_skew
accuracy_randdiag .= 2 .^ accuracy_randdiag

P = plot(framestyle=:box, xscale=:log2, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft, xticks = [32, 128, 512, 2048, 8192])
#plot!(P, 2N, accuracy_symplectoschur_sign, label="Algorithm 6.1 with " *L" -i\mathrm{sign}(iX)", marker=:diamond, linestyle=:solid, color=:green)
plot!(P, 2N, accuracy_symplectoschur_skew, label="Algorithm 6.1", marker=:circle, linestyle=:dash, color=:blue)
plot!(P, 2N, accuracy_schur, label="RSD with "*L"\texttt{schur}\ (\texttt{gees})", marker=:utriangle, linestyle=:dot, color=:purple)
plot!(P, 2N, accuracy_randdiag, label="EVD with " *L"\texttt{RandDiag}", marker=:star5, linestyle=:dashdot, color=:orange)
plot!(P, 2N[1:end], 10 * sqrt.(2N[1:end]) * eps(Float64) .* ones(length(2N[1:end])), label=L"\mathcal{O}(\epsilon_\mathrm{m}\sqrt{n})", marker=false, linestyle=:dashdot, color=:black)
plot!(P, 2N[1:end], 3* 2N[1:end].^2 * eps(Float64) .* ones(length(2N[1:end])), label=L"\mathcal{O}(\epsilon_\mathrm{m}n^2)", marker=false, linestyle=:dashdotdot, color=:black)
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, L"\Vert XQ - QS\ \Vert_\mathrm{F} / \Vert X\ \Vert_\mathrm{F}")
ylims!(P,  6 * 1e-16, 5 * 1e-9)
display(P)
savefig(P, joinpath(@__DIR__, "../figures/symplectoschur_accuracy.pdf"))

