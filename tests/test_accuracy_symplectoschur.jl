using LinearAlgebra, Plots, LaTeXStrings
include("../src/symplectoschur.jl")

BLAS.set_num_threads(4)
N = floor.(Int, 10 .^ LinRange(log10(100), log10(1000), 5)) .÷ 2
accuracy_schur = zeros(length(N))
accuracy_symplectoschur_sign = zeros(length(N))
accuracy_symplectoschur_skew = zeros(length(N))
Ntrials = 100
for (i, n) ∈ enumerate(N)
    for _ ∈ 1:Ntrials
        X = Matrix(qr(randn(2n, 2n)).Q)
        if det(X) < 0
            X[:,1] .= -X[:,1]
        end
        SchurX = schur(X)
        accuracy_schur[i] += log2(norm(X * SchurX.Z - SchurX.Z * SchurX.T) / norm(X)) / Ntrials 
        Q, C, S = symplectoschur!(copy(X); skew = false, polar = :hybrid)
        accuracy_symplectoschur_sign[i] += log2(norm(X * Q - Q * [C -S; S C]) / norm(X)) / Ntrials
        Q, C, S = symplectoschur!(copy(X); skew = true, polar = :hybrid)
        accuracy_symplectoschur_skew[i] += log2(norm(X * Q - Q * [C -S; S C]) / norm(X)) / Ntrials
    end 
end

accuracy_schur .= 2 .^ accuracy_schur
accuracy_symplectoschur_sign .= 2 .^ accuracy_symplectoschur_sign
accuracy_symplectoschur_skew .= 2 .^ accuracy_symplectoschur_skew

P = plot(framestyle=:box, xscale=:log10, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft)
plot!(P, 2N, accuracy_symplectoschur_sign, label="Algorithm 6.1 with " *L" -i\mathrm{sign}(iX)", marker=:diamond, linestyle=:solid, color=:green)
plot!(P, 2N, accuracy_symplectoschur_skew, label="Algorithm 6.1 with " *L"\mathrm{skopf}(\mathrm{skew}(X))", marker=:circle, linestyle=:dash, color=:blue)
plot!(P, 2N, accuracy_schur, label="RSD with "*L"\texttt{schur}\ (\texttt{gees})", marker=:utriangle, linestyle=:dot, color=:purple)
plot!(P, 2N, 10 * sqrt.(2N) * eps(Float64) .* ones(length(2N)), label=L"\mathcal{O}(\epsilon_\mathrm{m}\sqrt{n})", marker=false, linestyle=:dashdot, color=:black)
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, L"\Vert XQ - QS\ \Vert_\mathrm{F} / \Vert X\ \Vert_\mathrm{F}")
ylims!(P,  3 * 1e-15, 5 * 1e-13)
display(P)
savefig(P, joinpath(@__DIR__, "./figures/symplectoschur_accuracy.pdf"))

