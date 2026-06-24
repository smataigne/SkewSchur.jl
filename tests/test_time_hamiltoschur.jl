using LinearAlgebra, Plots, BenchmarkTools, LaTeXStrings
include("hamiltoschur.jl")
include("skew_schur.jl")
#=
BLAS.set_num_threads(6)
n = 10
A = randn(2n, 2n)
A = (A - A') / 2
@elapsed hamiltoschur!(copy(A); random = true, polar= :chol)
@elapsed hamiltoschur!(copy(A); random = true, polar= :qr)
@elapsed hamiltoschur!(copy(A); random = true, polar= :svd)
@elapsed householder_schur(copy(A))
@elapsed hermitian_eigen(copy(A))
N = floor.(Int, 10 .^ LinRange(log10(100), log10(1000), 5)) .÷ 2
times_hamiltoschur_qr = ones(length(N))
times_hamiltoschur_svd = ones(length(N))
times_hamiltoschur_chol = ones(length(N))
times_hamiltoschur_hybrid = ones(length(N))
times_householder_schur = ones(length(N))
times_hermitian_eigen = ones(length(N))
times_lapack_schur = ones(length(N))
times_hamiltoschur_chol .= Inf
times_hamiltoschur_qr .= Inf
times_hamiltoschur_svd .= Inf
times_hamiltoschur_hybrid .= Inf
times_householder_schur .= Inf
times_hermitian_eigen .= Inf
times_lapack_schur .= Inf
Ntrials = 10
for (i, n) ∈ enumerate(N)
    for trial in 1:Ntrials
        A = randn(2n, 2n)
        A = (A - A') / 2
        #times_hamiltoschur_chol[i] = min(times_hamiltoschur_chol[i], @elapsed hamiltoschur!(copy(A); random = true, polar= :chol))
        #times_hamiltoschur_qr[i] = min(times_hamiltoschur_qr[i], @elapsed hamiltoschur!(copy(A); random = true, polar= :qr))
        times_hamiltoschur_svd[i] = min(times_hamiltoschur_svd[i], @elapsed hamiltoschur!(copy(A); random = true, polar= :svd))
        times_hamiltoschur_hybrid[i] = min(times_hamiltoschur_hybrid[i], @elapsed hamiltoschur!(copy(A); random = true, polar= :hybrid))
        times_householder_schur[i] = min(times_householder_schur[i], @elapsed householder_schur(copy(A)))
        times_hermitian_eigen[i] = min(times_hermitian_eigen[i], @elapsed hermitian_eigen(copy(A)))
        times_lapack_schur[i] = min(times_lapack_schur[i], @elapsed schur(copy(A)))
    end
    println("Done with n = $n")
end
=#
P = plot(framestyle=:box, xscale=:log10, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft)
#plot!(P, 2N, times_hamiltoschur_qr, label="Algorithm 4.1 with QDWH", marker=:circle, linestyle=:solid, color=:blue)
#plot!(P, 2N, times_hamiltoschur_chol, label="Algorithm 4.1 with CDWH", marker=:square, linestyle=:dash, color=:green)
plot!(P, 2N, times_hamiltoschur_svd, label="Algorithm 4.1 with PSVD", marker=:pentagon, linestyle=:solid, color=:blue)
plot!(P, 2N, times_hamiltoschur_hybrid, label="Algorithm 4.1 with Q-CDWH", marker=:diamond, linestyle=:dash, color=:green)
plot!(P, 2N, times_householder_schur, label="RSD with tridiagonalization", marker=:dtriangle, linestyle=:dashdot, color=:red)
plot!(P, 2N, times_lapack_schur, label="RSD with "*L"\texttt{schur}\ (\texttt{gees})", marker=:hexagon, linestyle=:solid, color=:purple)
plot!(P, 2N, times_hermitian_eigen, label="EVD of " * L"iA", marker=:diamond, linestyle=:dashdotdot, color=:orange)

xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, "Time (s)")
script_dir = @__DIR__
path = joinpath(script_dir, "./figures/hamiltoschur_time_2.pdf")
savefig(P, path)
display(P)

P = plot(framestyle=:box, xscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft)
#plot!(P, 2N, times_hamiltoschur_qr, label="Algorithm 4.1 with QDWH", marker=:circle, linestyle=:solid, color=:blue)
#plot!(P, 2N, times_hamiltoschur_chol, label="Algorithm 4.1 with CDWH", marker=:square, linestyle=:dash, color=:green)
plot!(P, 2N, times_hamiltoschur_svd ./ times_householder_schur, label="Algorithm 4.1 with PSVD", marker=:pentagon, linestyle=:solid, color=:blue)
plot!(P, 2N, times_hamiltoschur_hybrid./ times_householder_schur, label="Algorithm 4.1 with Q-CDWH", marker=:diamond, linestyle=:dash, color=:green)
plot!(P, 2N, times_householder_schur ./ times_householder_schur, label="RSD with tridiagonalization", marker=:dtriangle, linestyle=:dashdot, color=:red)
plot!(P, 2N, times_lapack_schur ./ times_householder_schur, label="RSD with "*L"\texttt{schur} \ (\texttt{gees})", marker=:hexagon, linestyle=:solid, color=:purple)
plot!(P, 2N, times_hermitian_eigen ./ times_householder_schur, label="EVD of " * L"iA", marker=:diamond, linestyle=:dashdotdot, color=:orange)

xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, "Relative time")
ylims!(0.5, 12.5)
script_dir = @__DIR__
path = joinpath(script_dir, "./figures/hamiltoschur_time_2_relative.pdf")
savefig(P, path)
display(P)
