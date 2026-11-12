using LinearAlgebra, Plots, BenchmarkTools, LaTeXStrings
include("../src/hamiltoschur.jl")
include("../src/other_schur.jl")

BLAS.set_num_threads(64)
xticks = floor.(Int, 2 .^ LinRange(log2(1024), log2(16384), 5))
N = xticks .÷ 2

#Choose type of matrices to test: "random", "bad", "good"
type = "random"

times_hamiltoschur_svd = ones(length(N))
times_hamiltoschur_newton = ones(length(N))
times_hamiltoschur_hybrid = ones(length(N))
times_householder_schur = ones(length(N))
times_hermitian_eigen = ones(length(N))
times_lapack_schur = ones(length(N))
times_hamiltoschur_newton .= Inf
times_hamiltoschur_svd .= Inf
times_hamiltoschur_hybrid .= Inf
times_householder_schur .= Inf
times_hermitian_eigen .= Inf
times_lapack_schur .= Inf
Ntrials = 1
for (i, n) ∈ enumerate(N)
    for trial in 1:Ntrials
        if type == "random"
            A = randn(2n, 2n)
            A = (A - A') / 2
        elseif type == "bad"
            V = qr(randn(2n, 2n)).Q
            Σ = exp2.(-LinRange(0, 52, n))
            S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
            A = V * S * V'
            A = (A - A') / 2
        elseif type == "good"
            V = qr(randn(2n, 2n)).Q
            Σ = 0.5 .+ rand(n)
            S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
            A = V * S * V'
            A = (A - A') / 2
        end
        times_hamiltoschur_svd[i] = min(times_hamiltoschur_svd[i], @elapsed hamiltoschur!(copy(A); polar= :svd))
        times_hamiltoschur_hybrid[i] = min(times_hamiltoschur_hybrid[i], @elapsed hamiltoschur!(copy(A); polar= :hybrid))
        if type == "good"
            times_hamiltoschur_newton[i] = min(times_hamiltoschur_newton[i], @elapsed hamiltoschur!(copy(A); polar= :newton))
        end
        times_householder_schur[i] = min(times_householder_schur[i], @elapsed householder_schur(copy(A)))
        times_hermitian_eigen[i] = min(times_hermitian_eigen[i], @elapsed hermitian_eigen(copy(A)))
        #times_lapack_schur[i] = min(times_lapack_schur[i], @elapsed schur(copy(A)))
    end
    println("Done with n = $n")
end

P = plot(framestyle=:box, xscale=:log2, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 11, labelfontsize = 14, tickfontsize = 11, legend=:topleft, xticks = xticks)

plot!(P, 2N, times_hamiltoschur_svd, label="Algorithm 4.1 with PSVD", marker=:circle, linestyle=:solid, color=:blue)
plot!(P, 2N, times_hamiltoschur_hybrid, label="Algorithm 4.1 with QDWH", marker=:pentagon, linestyle=:dash, color=:green)
plot!(P, 2N, times_hamiltoschur_newton, label="Algorithm 4.1 with Newton-Schulz", marker=:square, linestyle=:dashdot, color=:purple)
plot!(P, 2N, times_householder_schur, label="RSD with tridiagonalization", marker=:dtriangle, linestyle=:dashdotdot, color=:red)
#plot!(P, 2N, times_lapack_schur, label="RSD with "*L"\texttt{schur}\ (\texttt{gees})", marker=:square, linestyle=:dashdotdot, color=:purple)
plot!(P, 2N, times_hermitian_eigen, label="EVD of " * L"\mathrm{i}A", marker=:diamond, linestyle=:dot, color=:orange)
plot!(P, 2N[2:end], 0.03 * 1e-8N[2:end].^3 , label=L"\mathcal{O}(n^3)", marker=false, linestyle=:dot, color=:black)
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, "Time (s)")
if type == "random"
    title!(P, "Random Gaussian matrices")
elseif type == "bad"
    title!(P, "Random ill-conditioned matrices")
elseif type == "good"
    title!(P, "Random well-conditioned matrices")
end
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/hamiltoschur_time_2_" * type * ".pdf")
pathpng = joinpath(script_dir, "../figures/hamiltoschur_time_2_" * type * ".png")
savefig(P, path)
savefig(P, pathpng)
display(P)

P = plot(framestyle=:box, xscale=:log2, xticks = xticks, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 11, labelfontsize = 14, tickfontsize = 11, legend=:topleft, yscale=:log2)
#plot!(P, 2N, times_hamiltoschur_qr, label="Algorithm 4.1 with QDWH", marker=:circle, linestyle=:solid, color=:blue)

plot!(P, 2N, times_hamiltoschur_svd ./ times_householder_schur, label="Algorithm 4.1 with PSVD", marker=:circle, linestyle=:solid, color=:blue)
plot!(P, 2N, times_hamiltoschur_hybrid./ times_householder_schur, label="Algorithm 4.1 with QDWH", marker=:pentagon, linestyle=:dash, color=:green)
plot!(P, 2N, times_hamiltoschur_newton ./ times_householder_schur, label="Algorithm 4.1 with Newton-Schulz", marker=:square, linestyle=:dashdot, color=:purple)
plot!(P, 2N, times_householder_schur ./ times_householder_schur, label="RSD with tridiagonalization", marker=:dtriangle, linestyle=:dashdotdot, color=:red)
#plot!(P, 2N, times_lapack_schur ./ times_householder_schur, label="RSD with "*L"\texttt{schur} \ (\texttt{gees})", marker=:square, linestyle=:dashdotdot, color=:purple)
plot!(P, 2N, times_hermitian_eigen ./ times_householder_schur, label="EVD of " * L"\mathrm{i}A", marker=:diamond, linestyle=:dot, color=:orange)

xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, "Relative time")
ylims!(P, 0.5, 8)
if type == "random"
    title!(P, "Random Gaussian matrices")
elseif type == "bad"
    title!(P, "Random ill-conditioned matrices")
elseif type == "good"
    title!(P, "Random well-conditioned matrices")
end
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/hamiltoschur_time_2_relative_" * type * ".pdf")
pathpng = joinpath(script_dir, "../figures/hamiltoschur_time_2_relative_" * type * ".png")
savefig(P, path)
savefig(P, pathpng)
display(P)


