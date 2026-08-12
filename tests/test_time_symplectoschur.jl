using LinearAlgebra, Plots, LaTeXStrings
include("../src/symplectoschur.jl")
include("../src/other_schur.jl")

BLAS.set_num_threads(6)
N = floor.(Int, 2 .^ LinRange(log2(8), log2(2048), 5)) .÷ 2
time_schur = zeros(length(N))
time_symplectoschur_sign = zeros(length(N))
time_symplectoschur_skew = zeros(length(N))
time_randdiag = zeros(length(N))
time_symplectoschur_sign .= Inf
time_symplectoschur_skew .= Inf
time_randdiag .= Inf
time_schur .= Inf
Ntrials = 10
for (i, n) ∈ enumerate(N)
    for _ ∈ 1:Ntrials
        X = Matrix(qr(randn(2n, 2n)).Q)
        if det(X) < 0
            X[:,1] .= -X[:,1]
        end
        time_schur[i] = min(time_schur[i], @elapsed schur(X))
        #time_symplectoschur_sign[i] = min(time_symplectoschur_sign[i], @elapsed symplectoschur!(copy(X); skew = false, polar = :hybrid))
        time_symplectoschur_skew[i] = min(time_symplectoschur_skew[i], @elapsed symplectoschur!(copy(X); skew = true, polar = :hybrid))
        time_randdiag[i] = min(time_randdiag[i], @elapsed randdiag(copy(X)))
    end
end

P = plot(framestyle=:box, xscale=:log2, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft, xticks = [8, 32, 128, 512, 2048])
#plot!(P, 2N, time_symplectoschur_sign, label="Algorithm 6.1 with " *L" -i\mathrm{sign}(iX)", marker=:diamond, linestyle=:solid, color=:green)
plot!(P, 2N, time_symplectoschur_skew, label = "Algorithm 6.1 with " *L"\mathrm{skopf}(\mathrm{skew}(X))", marker=:circle, linestyle=:dash, color=:blue)
plot!(P, 2N, time_schur, label="RSD with "*L"\texttt{schur}\ (\texttt{gees})", marker=:utriangle, linestyle=:dot, color=:purple)
plot!(P, 2N, time_randdiag, label="EVD with " *L"\texttt{RandDiag}", marker=:star5, linestyle=:dashdot, color=:orange)
plot!(P, 2N[2:end],  0.1 * 1e-8N[2:end].^3 , label=L"\mathcal{O}(n^3)", marker=false, linestyle=:dash, color=:black)
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, "Time (s)")
ylims!(P,  1e-6, 100)
display(P)
#savefig(P, joinpath(@__DIR__, "../figures/symplectoschur_time.pdf"))
println("Done with symplectoschur time tests")
