using LinearAlgebra, Plots, Distributions, LaTeXStrings

T = 50
N = floor.(Int, exp10.(LinRange(1, 3, 8))) .÷ 2
lc = zeros(length(N), T)
lc2 = zeros(length(N), T)
for (i, n) ∈ enumerate(N)
    Q = qr(randn(2n, 2n)).Q
    J = [zeros(n, n) -Diagonal(1:n); Diagonal(1:n) zeros(n, n)]
    P = Q * J * Q'
    for t ∈ 1:T
        V = randn(2n, n)
        lc[i, t] = cond(V)
        lc2[i, t] = cond(complex.(P * V, V))
    end
end

P = plot(framestyle=:box, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft)
plot!(2N, mean(lc, dims=2)[:], xaxis = :log10, marker=:o, label = L"\kappa(V)", color=:blue)
plot!(2N, mean(lc2, dims=2)[:], xaxis = :log10, label = L"\kappa((P+iI)V)", color=:green, linestyle=:dash, marker=:diamond)
lower = mean(lc, dims=2)[:].-2*std(lc, dims=2)[:]
upper = mean(lc, dims=2)[:].+2*std(lc, dims=2)[:]
lower2 = mean(lc2, dims=2)[:].-2*std(lc2, dims=2)[:]
upper2 = mean(lc2, dims=2)[:].+2*std(lc2, dims=2)[:]
plot!(2N, lower, fillrange=upper, fillalpha=0.25,
      label=false, linecolor=:transparent, color=:lightblue)
plot!(2N, lower2, fillrange=upper2, fillalpha=0.25,
      label=false, linecolor=:transparent, color=:lightgreen)
plot!(2N, ones(size(2N)) .* (2+sqrt(2))/(2 - sqrt(2)), label=L"(2+\sqrt{2})/(2 - \sqrt{2})", marker=false, linestyle=:dashdot, color=:black)
xlabel!("Matrix size " * L"2n")
ylabel!("Condition number")
ylims!(3, 15)
display(P)
savefig(P, joinpath(@__DIR__, "./figures/conditioning_randn.pdf"))