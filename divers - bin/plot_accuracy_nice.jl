using LinearAlgebra, Plots, BenchmarkTools, LaTeXStrings
include("hamiltoschur.jl")

N = floor.(Int, 10 .^ LinRange(log10(10), log10(1000), 10)) .÷ 2
l_unsafe = zeros(length(N))
l_safe = zeros(length(N))
conditionings = zeros(length(N))
Ntrials = 20
for (i, n) ∈ enumerate(N)
    for trial in 1:Ntrials
        V = qr(randn(2n, 2n)).Q
        Σ = 1 .+ rand(n)
        S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
        A = V * S * V'
        A = (A - A') / 2
        conditionings[i] += log2(cond(A)) / Ntrials
        Q, vals = hamiltoschur!(copy(A); safe = false, polar = 1)
        S = [zeros(n, n) -Diagonal(vals); Diagonal(vals) zeros(n, n)]
        l_unsafe[i] += log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
        Q, vals = hamiltoschur!(copy(A); safe = true, polar = 1)
        S = [zeros(n, n) -Diagonal(vals); Diagonal(vals) zeros(n, n)]
        l_safe[i] += log2(norm(A * Q - Q * S)/ norm(A)) / Ntrials
    end
end
l_unsafe .= 2 .^l_unsafe
l_safe .= 2 .^l_safe
conditionings .= 2 .^conditionings

P = plot(framestyle=:box, xscale=:log10, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12)
plot!(P, 2N, l_unsafe, label=L"V = [I\ ;\  0]", marker=:circle, linestyle=:solid, color=:blue)
plot!(P, 2N, l_safe, label=L"V=\texttt{randn}(2n, n)", marker=:rect, linestyle=:dash, color=:red)
plot!(P, 2N, 2N * eps(Float64) * 10, label= L"\mathcal{O}(n \cdot \varepsilon_\mathrm{m}) ", linestyle=:dot, color =:black)
plot!(P, 2N, conditionings * eps(Float64), marker = :diamond, label= L"\kappa(A) \cdot \varepsilon_\mathrm{m} ", linestyle=:dashdot, color=:green)
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, L"\Vert A Q - Q S\ \Vert_\mathrm{F} / \Vert A\ \Vert_\mathrm{F}")
script_dir = @__DIR__
path = joinpath(script_dir, "./figures/hamiltoschur_accuracy_good_k_qr.pdf")
savefig(P, path)
display(P)