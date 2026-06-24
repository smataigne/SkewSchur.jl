using LinearAlgebra, Plots, BenchmarkTools, LaTeXStrings
include("hamiltoschur.jl")
include("skew_schur.jl")
#=

N = floor.(Int, 10 .^ LinRange(log10(10), log10(1000), 10)) .÷ 2
errors_bad_qr = zeros(length(N))
errors_random_qr = zeros(length(N))
errors_nice_qr = zeros(length(N))
errors_bad_chol = zeros(length(N))
errors_random_chol = zeros(length(N))
errors_nice_chol = zeros(length(N))
errors_bad_householder = zeros(length(N))
errors_random_householder = zeros(length(N))
errors_nice_householder = zeros(length(N))
errors_bad_hermitian = zeros(length(N))
errors_random_hermitian = zeros(length(N))
errors_nice_hermitian = zeros(length(N))
errors_bad_schur = zeros(length(N))
errors_random_schur = zeros(length(N))
errors_nice_schur = zeros(length(N))
conditionings= zeros(length(N))
Ntrials = 10
for (i, n) ∈ enumerate(N)
    for trial in 1:Ntrials
        #bad case
        V = qr(randn(2n, 2n)).Q
        Σ = exp2.(-LinRange(0, 52, n))
        S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
        A = V * S * V'
        A = (A - A') / 2
        conditionings[i] += log2(cond(A)) / Ntrials
        Q, vals = hamiltoschur!(copy(A); random = true, polar = :qr)
        S = [zeros(n, n) -Diagonal(vals); Diagonal(vals) zeros(n, n)]
        errors_bad_qr[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, vals = hamiltoschur!(copy(A); random = true, polar = :hybrid)
        S = [zeros(n, n) -Diagonal(vals); Diagonal(vals) zeros(n, n)]
        errors_bad_chol[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, Σ= householder_schur(copy(A))
        S = [zeros(n, n) -Σ; Σ zeros(n, n)]
        errors_bad_householder[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, Λ = hermitian_eigen(copy(A))
        errors_bad_hermitian[i] +=  log2(norm(A * Q - Q * Λ)/ norm(A)) / Ntrials
        Schur = schur(A)
        errors_bad_schur[i] += log2(norm(A * Schur.Z - Schur.Z * Schur.T)/ norm(A)) / Ntrials

        #random case
        A = randn(2n, 2n)
        A = (A - A') / 2
        Q, vals = hamiltoschur!(copy(A); random = true, polar = :qr)
        S = [zeros(n, n) -Diagonal(vals); Diagonal(vals) zeros(n, n)]
        errors_random_qr[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, vals = hamiltoschur!(copy(A); random = true, polar = :hybrid)
        S = [zeros(n, n) -Diagonal(vals); Diagonal(vals) zeros(n, n)]
        errors_random_chol[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, Σ = householder_schur(copy(A))
        S = [zeros(n, n) -Σ; Σ zeros(n, n)]
        errors_random_householder[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, Λ = hermitian_eigen(copy(A))
        errors_random_hermitian[i] +=  log2(norm(A * Q - Q * Λ)/ norm(A)) / Ntrials
        Schur = schur(A)
        errors_random_schur[i] += log2(norm(A * Schur.Z - Schur.Z * Schur.T)/ norm(A)) / Ntrials
        #nice case
        V = qr(randn(2n, 2n)).Q
        Σ = 1 .+ rand(n)
        S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
        A = V * S * V'
        A = (A - A') / 2
        Q, vals = hamiltoschur!(copy(A); random = true, polar = :qr)
        S = [zeros(n, n) -Diagonal(vals); Diagonal(vals) zeros(n, n)]
        errors_nice_qr[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, vals = hamiltoschur!(copy(A); random = true, polar = :hybrid)
        S = [zeros(n, n) -Diagonal(vals); Diagonal(vals) zeros(n, n)]
        errors_nice_chol[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, Σ = householder_schur(copy(A))
        S = [zeros(n, n) -Σ; Σ zeros(n, n)]
        errors_nice_householder[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, Λ = hermitian_eigen(copy(A))
        errors_nice_hermitian[i] +=  log2(norm(A * Q - Q * Λ)/ norm(A)) / Ntrials
        Schur = schur(A)
        errors_nice_schur[i] += log2(norm(A * Schur.Z - Schur.Z * Schur.T)/ norm(A)) / Ntrials
    end
end
errors_bad_qr .= 2 .^errors_bad_qr
errors_random_qr .= 2 .^errors_random_qr
errors_nice_qr .= 2 .^errors_nice_qr
errors_bad_chol .= 2 .^errors_bad_chol
errors_random_chol .= 2 .^errors_random_chol
errors_nice_chol .= 2 .^errors_nice_chol
errors_bad_householder .= 2 .^errors_bad_householder
errors_random_householder .= 2 .^errors_random_householder
errors_nice_householder .= 2 .^errors_nice_householder
errors_bad_hermitian .= 2 .^errors_bad_hermitian
errors_random_hermitian .= 2 .^errors_random_hermitian
errors_nice_hermitian .= 2 .^errors_nice_hermitian
errors_bad_schur .= 2 .^errors_bad_schur
errors_random_schur .= 2 .^errors_random_schur
errors_nice_schur .= 2 .^errors_nice_schur
conditionings .= 2 .^conditionings
=#
P = plot(framestyle=:box, xscale=:log10, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft)
plot!(P, 2N, errors_random_qr, label="Algorithm 4.1 with QDWH", marker=:circle, linestyle=:solid, color=:blue)
plot!(P, 2N, errors_random_chol, label="Algorithm 4.1 with Q-CDWH", marker=:square, linestyle=:solid, color=:green)
plot!(P, 2N, errors_random_householder, label="RSD with tridiagonalization", marker=:dtriangle, linestyle=:solid, color=:red)
plot!(P, 2N, errors_random_schur, label="RSD with " *L" \texttt{schur} (\texttt{gees})", marker=:cross, linestyle=:solid, color=:purple)
plot!(P, 2N, errors_random_hermitian, label="EVD of " * L"iA", marker=:diamond, linestyle=:solid, color=:orange)
plot!(P, 2N, 2N * eps(Float64) * 10, label= L"\mathcal{O}(n \cdot \varepsilon_\mathrm{m}) ", linestyle=:dot, color =:black)
title!("Random matrices")
ylims!(P, 1e-16, 1e-10)
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, L"\Vert AQ - QS\ \Vert_\mathrm{F} / \Vert A\ \Vert_\mathrm{F}")
script_dir = @__DIR__
path = joinpath(script_dir, "./figures/hamiltoschur_accuracy_random.pdf")
savefig(P, path)
display(P)

P = plot(framestyle=:box, xscale=:log10, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft)
plot!(P, 2N, errors_bad_qr, label="Algorithm 4.1 with QDWH", marker=:circle, linestyle=:solid, color=:blue)
plot!(P, 2N, errors_bad_chol, label="Algorithm 4.1 with Q-CDWH", marker=:square, linestyle=:solid, color=:green)
plot!(P, 2N, errors_bad_householder, label="RSD with tridiagonalization", marker=:dtriangle, linestyle=:solid, color=:red)
plot!(P, 2N, errors_bad_schur, label="RSD with " *L" \texttt{schur} (\texttt{gees})", marker=:cross, linestyle=:solid, color=:purple)
plot!(P, 2N, errors_bad_hermitian, label="EVD of " * L"iA", marker=:diamond, linestyle=:solid, color=:orange)
plot!(P, 2N, 2N * eps(Float64) * 10, label= L"\mathcal{O}(n \cdot \varepsilon_\mathrm{m}) ", linestyle=:dot, color =:black)
title!("Matrices with " * L"\kappa(A) = \mathcal{O}(\varepsilon_\mathrm{m}^{-1})")
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, L"\Vert AQ - QS\ \Vert_\mathrm{F} / \Vert A\ \Vert_\mathrm{F}")
ylims!(P, 1e-16, 1e-10)
script_dir = @__DIR__
path = joinpath(script_dir, "./figures/hamiltoschur_accuracy_bad.pdf")
savefig(P, path)
display(P)

P = plot(framestyle=:box, xscale=:log10, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft)
plot!(P, 2N, errors_nice_qr, label="Algorithm 4.1 with QDWH", marker=:circle, linestyle=:solid, color=:blue)
plot!(P, 2N, errors_nice_chol, label="Algorithm 4.1 with Q-CDWH", marker=:square, linestyle=:solid, color=:green)
plot!(P, 2N, errors_nice_householder, label="RSD with tridiagonalization", marker=:dtriangle, linestyle=:solid, color=:red)
plot!(P, 2N, errors_nice_schur, label="RSD with " *L" \texttt{schur} (\texttt{gees})", marker=:cross, linestyle=:solid, color=:purple)
plot!(P, 2N, errors_nice_hermitian, label="EVD of " * L"iA", marker=:diamond, linestyle=:solid, color=:orange)
plot!(P, 2N, 2N * eps(Float64), label= L"\mathcal{O}(n \cdot \varepsilon_\mathrm{m}) ", linestyle=:dot, color =:black)
title!("Matrices with " * L"\kappa(A) = \mathcal{O}(1)")
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, L"\Vert AQ - QS\ \Vert_\mathrm{F} / \Vert A\ \Vert_\mathrm{F}")
ylims!(P, 5 * 1e-16, 1e-12)
script_dir = @__DIR__
path = joinpath(script_dir, "./figures/hamiltoschur_accuracy_nice.pdf")
savefig(P, path)
display(P)