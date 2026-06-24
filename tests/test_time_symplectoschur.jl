using LinearAlgebra, Plots, LaTeXStrings
include("symplectoschur.jl")

BLAS.set_num_threads(4)
N = floor.(Int, 10 .^ LinRange(log10(100), log10(1000), 5)) .÷ 2
time_schur = zeros(length(N))
time_symplectoschur_sign = zeros(length(N))
time_symplectoschur_skew = zeros(length(N))
time_symplectoschur_sign .= Inf
time_symplectoschur_skew .= Inf
time_schur .= Inf
Ntrials = 10
for (i, n) ∈ enumerate(N)
    for _ ∈ 1:Ntrials
        X = Matrix(qr(randn(2n, 2n)).Q)
        if det(X) < 0
            X[:,1] .= -X[:,1]
        end
        time_schur[i] = min(time_schur[i], @elapsed schur(X))
        time_symplectoschur_sign[i] = min(time_symplectoschur_sign[i], @elapsed symplectoschur!(copy(X); skew = false, polar = :hybrid))
        time_symplectoschur_skew[i] = min(time_symplectoschur_skew[i], @elapsed symplectoschur!(copy(X); skew = true, polar = :hybrid))
    end
end

P = plot(framestyle=:box, xscale=:log10, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft)
plot!(P, 2N, time_symplectoschur_sign, label="Algorithm 6.1 with " *L" -i\mathrm{sign}(iX)", marker=:diamond, linestyle=:solid, color=:green)
plot!(P, 2N, time_symplectoschur_skew, label="Algorithm 6.1 with " *L"\mathrm{skopf}(\mathrm{skew}(X))", marker=:circle, linestyle=:dash, color=:blue)
plot!(P, 2N, time_schur, label="RSD with "*L"\texttt{schur}\ (\texttt{gees})", marker=:utriangle, linestyle=:dot, color=:purple)
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, "Time (s)")
display(P)
savefig(P, joinpath(@__DIR__, "./figures/symplectoschur_time.pdf"))
P = plot(framestyle=:box, xscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft)
plot!(P, 2N, time_symplectoschur_sign ./ time_schur, label="Algorithm 6.1 with " *L" -i\mathrm{sign}(iX)", marker=:diamond, linestyle=:solid, color=:green)
plot!(P, 2N, time_symplectoschur_skew ./ time_schur, label="Algorithm 6.1 with " *L"\mathrm{skopf}(\mathrm{skew}(X))", marker=:circle, linestyle=:dash, color=:blue)
plot!(P, 2N, time_schur ./ time_schur, label="RSD with "*L"\texttt{schur}\ (\texttt{gees})", marker=:utriangle, linestyle=:dot, color=:purple)
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, "Relative time")
savefig(P, joinpath(@__DIR__, "./figures/symplectoschur_time_relative.pdf"))