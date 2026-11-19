using LinearAlgebra, Plots, BenchmarkTools, LaTeXStrings
include("../src/hamiltoschur.jl")
include("../src/other_schur.jl")
#=
nthreads = 128
BLAS.set_num_threads(nthreads)
xticks = floor.(Int, 2 .^ LinRange(log2(1024), log2(32768), 6))
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
=#
Ntrials = 2
for (i, n) ∈ enumerate(N)
    if i < 2
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
        if type != "bad"
            #SVD doesn't work for ill-conditioned matrices, so we skip it for those cases. Memory error for large matrices, so we skip it for those cases as well.
            times_hamiltoschur_svd[i] = min(times_hamiltoschur_svd[i], @elapsed hamiltoschur!(copy(A); polar= :svd))
            println("Done with svd")
        end
        if 2n < 20000
            times_hamiltoschur_hybrid[i] = min(times_hamiltoschur_hybrid[i], @elapsed hamiltoschur!(copy(A); polar= :hybrid))
            println("Done with hybrid")
        else
            times_hamiltoschur_hybrid[i] = min(times_hamiltoschur_hybrid[i], @elapsed hamiltoschur!(copy(A); polar= :chol))
            println("Done with hybrid")
        end
        if type == "good"
            times_hamiltoschur_newton[i] = min(times_hamiltoschur_newton[i], @elapsed hamiltoschur!(copy(A); polar= :newton))
            println("Done with newton")
        else
            if 2n < 20000  #Hardware constraints do not allow the use of the QDWH for large matrices, so we use the Cholesky-based method instead.
                times_hamiltoschur_newton[i] = min(times_hamiltoschur_newton[i], @elapsed hamiltoschur!(copy(A); polar= :newtonsafeqr))
            println("Done with newtonsafeqr")
            else
                times_hamiltoschur_newton[i] = min(times_hamiltoschur_newton[i], @elapsed hamiltoschur!(copy(A); polar= :newtonsafechol))
            println("Done with newtonsafechol")
            end
        end
        times_householder_schur[i] = min(times_householder_schur[i], @elapsed householder_schur(copy(A)))
        println("Done with householder")
        if type == "bad" && 2n > 4200
            println("Skipping hermitian eigen for bad matrices of size $(2n)")
        else
            times_hermitian_eigen[i] = min(times_hermitian_eigen[i], @elapsed hermitian_eigen(copy(A)))
        end
        println("Done with hermitian")
    end
    println("Done with n = $n")
end
end

P = plot(framestyle=:box, xscale=:log2, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 11, labelfontsize = 14, tickfontsize = 11, legend=:topleft, xticks = xticks)
if type != "bad"
    plot!(P, 2N, times_hamiltoschur_svd, label="Algorithm 4.1 with PSVD", marker=:circle, linestyle=:solid, color=:blue)
end
plot!(P, 2N, times_hamiltoschur_hybrid, label="Algorithm 4.1 with QDWH", marker=:pentagon, linestyle=:dash, color=:green)
plot!(P, 2N, times_hamiltoschur_newton, label="Algorithm 4.1 with Newton-Schulz", marker=:square, linestyle=:dashdot, color=:purple)
plot!(P, 2N, times_householder_schur, label="RSD with tridiagonalization", marker=:dtriangle, linestyle=:dashdotdot, color=:red)
plot!(P, 2N, times_hermitian_eigen, label="EVD of " * L"\mathrm{i}A", marker=:diamond, linestyle=:dot, color=:orange)
plot!(P, 2N[2:end], 0.015 * 1e-8N[2:end].^3 , label=L"\mathcal{O}(n^3)", marker=false, linestyle=:dot, color=:black)
ylims!(P, 1e-1, 2*1e4)
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
path = joinpath(script_dir, "../figures/hamiltoschur_time_2_" * type * "_nthreads_" * string(nthreads) * ".pdf")
pathpng = joinpath(script_dir, "../figures/hamiltoschur_time_2_" * type * "_nthreads_" * string(nthreads) * ".png")
savefig(P, path)
savefig(P, pathpng)

P = plot(framestyle=:box, xscale=:log2, xticks = xticks, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 11, labelfontsize = 14, tickfontsize = 11, legend=:topright, yscale=:log2)
if type != "bad"
    plot!(P, 2N, times_hamiltoschur_svd ./ times_householder_schur, label="Algorithm 4.1 with PSVD", marker=:circle, linestyle=:solid, color=:blue)
end
plot!(P, 2N, times_hamiltoschur_hybrid./ times_householder_schur, label="Algorithm 4.1 with QDWH", marker=:pentagon, linestyle=:dash, color=:green)
plot!(P, 2N, times_hamiltoschur_newton ./ times_householder_schur, label="Algorithm 4.1 with Newton-Schulz", marker=:square, linestyle=:dashdot, color=:purple)
plot!(P, 2N, times_householder_schur ./ times_householder_schur, label="RSD with tridiagonalization", marker=:dtriangle, linestyle=:dashdotdot, color=:red)
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
path = joinpath(script_dir, "../figures/hamiltoschur_time_2_relative_" * type * "_nthreads_" * string(nthreads) * ".pdf")
pathpng = joinpath(script_dir, "../figures/hamiltoschur_time_2_relative_" * type * "_nthreads_" * string(nthreads) * ".png")
savefig(P, path)
savefig(P, pathpng)
