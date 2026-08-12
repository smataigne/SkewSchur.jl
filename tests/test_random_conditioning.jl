using LinearAlgebra, Plots, Distributions, LaTeXStrings, SkewLinearAlgebra

T = 10
N = floor.(Int, exp2.(LinRange(log2(16), log2(4096), 7))) .÷ 2
lc = zeros(length(N), T)
lc2 = zeros(length(N), T)
for (i, n) ∈ enumerate(N)
    Q = qr(randn(2n, 2n)).Q
    J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
    P = Q * J * Q'
    v = abs.(randn(n))
    A = Q * [zeros(n, n) -Diagonal(v); Diagonal(v) zeros(n, n)] * Q'
    for t ∈ 1:T
        V = randn(2n, n)
        lc[i, t] = cond(V)
        lc2[i, t] = cond((P+1im*I(2n)) * V)
    end
end
P = plot(framestyle=:box, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft, yaxis=:log10, xticks = [16, 64, 256, 1024, 4096])
plot!(2N, mean(lc, dims=2)[:], xaxis = :log2, marker=:o, label = L"\kappa_2(V)", color=:blue)
plot!(2N, mean(lc2, dims=2)[:], xaxis = :log2, label = L"\kappa_2((P+iI)V)", color=:green, linestyle=:dash, marker=:diamond)
xlabel!("Matrix size " * L"2n")
ylabel!("Condition number")
display(P)
#savefig(P, joinpath(@__DIR__, "../figures/conditioning_randn.pdf"))