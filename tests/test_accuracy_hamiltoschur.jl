using LinearAlgebra, Plots, BenchmarkTools, LaTeXStrings
include("../src/hamiltoschur.jl")
include("../src/other_schur.jl")

BLAS.set_num_threads(16)
N = floor.(Int, 2 .^ LinRange(log2(16), log2(4096), 6)) .÷ 2
#Choose type of matrices to test: "random", "bad", "good"
type = "bad"
errors_hybrid_1 = zeros(length(N))
errors_hybrid_2 = zeros(length(N))
errors_svd = zeros(length(N))
errors_schur= zeros(length(N))
errors_hermitian = zeros(length(N))
Ntrials = 2
for (i, n) ∈ enumerate(N)
    for trial in 1:Ntrials
        if type == "good"
            V = qr(randn(2n, 2n)).Q
            Σ = 0.5 .+ rand(n)
            S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
            A = V * S * V'
            A = (A - A') / 2
        elseif type == "random"
            A = randn(2n, 2n)
            A = (A - A') / 2
        elseif type == "bad"
            V = qr(randn(2n, 2n)).Q
            Σ = exp2.(-LinRange(0, 52, n))
            S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
            A = V * S * V'
            A = (A - A') / 2
        end
        #=
        Q, Σ = hamiltoschur!(copy(A); polar = :svd)
        S = [zeros(n, n) -Σ; Σ zeros(n, n)]
        errors_svd[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        =#
        Q, Σ = hamiltoschur!(copy(A); polar = :hybrid, subspace = 1)
        S = [zeros(n, n) -Σ; Σ zeros(n, n)]
        errors_hybrid_1[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, Σ = hamiltoschur!(copy(A); polar = :hybrid, subspace = 2)
        S = [zeros(n, n) -Σ; Σ zeros(n, n)]
        errors_hybrid_2[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, Σ = householder_schur(copy(A))
        S = [zeros(n, n) -Σ; Σ zeros(n, n)]
        errors_schur[i] +=  log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, Λ = hermitian_eigen(copy(A))
        errors_hermitian[i] +=  log2(norm(A * Q - Q * Λ)/ norm(A)) / Ntrials
    end
end
errors_hybrid_1 .= 2 .^errors_hybrid_1
errors_hybrid_2 .= 2 .^errors_hybrid_2
errors_svd .= 2 .^errors_svd
errors_schur .= 2 .^errors_schur
errors_hermitian .= 2 .^errors_hermitian

P = plot(framestyle=:box, xscale=:log2, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft, xticks = [16, 64, 256, 1024, 4096])
#plot!(P, 2N, errors_svd, label="Algorithm 4.1 with PSVD", marker=:circle, linestyle=:solid, color=:blue)
plot!(P, 2N, errors_hybrid_1, label="Algorithm 4.1 with QDWH (1 subspace it.)", marker=:pentagon, linestyle=:dash, color=:green)
plot!(P, 2N, errors_hybrid_2, label="Algorithm 4.1 with QDWH (2 subspace it.)", marker=:square, linestyle=:dashdot, color=:purple)
plot!(P, 2N, errors_schur, label="RSD with tridiagonalization", marker=:dtriangle, linestyle=:dashdotdot, color=:red)
plot!(P, 2N, errors_hermitian, label="EVD of " * L"\mathrm{i}A", marker=:diamond, linestyle=:dot, color=:orange)
plot!(P, 2N, 2N * eps(Float64) * 3, label= L"\mathcal{O}(n \cdot \varepsilon_\mathrm{m}) ", linestyle=:dot, color =:black)
if type == "good"
    title!("Random well-conditioned matrices")
elseif type == "random"
    title!("Random Gaussian matrices")
elseif type == "bad"
    title!("Random ill-conditioned matrices")
end
ylims!(P, 1e-16, 1e-10)
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, L"\Vert AQ - QS\ \Vert_\mathrm{F} / \Vert A\ \Vert_\mathrm{F}")
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/hamiltoschur_accuracy_"*type*".pdf")
savefig(P, path)
display(P)

