using LinearAlgebra, Plots, Distributions, LaTeXStrings, SkewLinearAlgebra

T = 10
N = floor.(Int, exp2.(LinRange(log2(16), log2(4096), 7))) .÷ 2
lc = zeros(length(N), T)
lc2 = zeros(length(N), T)
lc3 = zeros(length(N), T)
for (i, n) ∈ enumerate(N)
    Q = qr(randn(2n, 2n)).Q
    J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
    P = Q * J * Q'
    v = abs.(randn(n))
    for t ∈ 1:T
        V = randn(2n, n)
        V2 = randn(ComplexF64, 2n, n)
        lc[i, t] = cond(V)
        lc2[i, t] = cond((P+1im*I(2n)) * V)
        lc3[i, t] = cond((P+1im*I(2n)) * Matrix(qr(((P+1im*I(2n))/2) * V).Q))
    end
end

P = plot(framestyle=:box, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft, yaxis=:log10, xticks = [16, 64, 256, 1024, 4096])
plot!(2N, mean(lc, dims=2)[:], xaxis = :log2, marker=:o, label = L"\kappa_2(V)", color=:blue, linestyle=:dashdot)
plot!(2N, mean(lc2, dims=2)[:], xaxis = :log2, label = L"\kappa_2((P+\mathrm{i}I)V)", color=:red, linestyle=:solid, marker=:circle)
plot!(2N, mean(lc3, dims=2)[:], xaxis = :log2, label = L"\kappa_2((P+\mathrm{i}I)\mathtt{qf}((P + \mathrm{i}I)V))", color=:green, linestyle=:dash, marker=:diamond)
xlabel!("Matrix size " * L"2n")
ylabel!("Condition number")
display(P)
savefig(P, joinpath(@__DIR__, "../figures/conditioning_randn.pdf"))