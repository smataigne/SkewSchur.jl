using LinearAlgebra, Plots, BenchmarkTools, LaTeXStrings, SkewLinearAlgebra
include("../src/polar.jl")

BLAS.set_num_threads(4)

@views function getM(P::AbstractMatrix{T}) where T
    n = size(P, 1) ÷ 2
    V = randn(T, 2n, n)
    temp = zeros(2n, 2n)
    mul!(temp[:, 1:n], P, V, 1, 0)
    U = complex.(temp[:, 1:n], V)
    M = Matrix(qr(U).Q)
    @. temp[:, 1:n] = √2 * real(M)        #reduce allocations
    @. temp[:, n+1:2n] = -√2 * imag(M)
    return temp
end

@views function dist_to_Hamiltonian(A::AbstractMatrix{T}) where T
    n = size(A, 1) ÷ 2
    Ah = zeros(T, 2n, 2n)
    @. Ah[1:n, 1:n] = (A[1:n, 1:n] + A[n+1:2n, n+1:2n]) / 2
    @. Ah[n+1:2n, 1:n] = (A[n+1:2n, 1:n] - A[1:n, n+1:2n]) / 2
    Ah[1:n, 1:n] .-= A[1:n, 1:n]'
    Ah[1:n, 1:n] ./= 2
    Ah[1:n, n+1:2n] .+= A[1:n, n+1:2n]'
    Ah[1:n, n+1:2n] ./= 2
    Ah[n+1:2n, n+1:2n] .= Ah[1:n, 1:n]
    Ah[1:n, n+1:2n] .= -Ah[n+1:2n, 1:n]
    return norm(Ah - A) / norm(A)
end

function dist_to_symplectic(A::AbstractMatrix{T}) where T
    n = size(A, 1) ÷ 2
    Y = (A[1:n, 1:n] + A[n+1:2n, n+1:2n]) / 2 + 1im * (A[n+1:2n, 1:n] - A[1:n, n+1:2n]) / 2
    Σ = Diagonal(svdvals(Y))
    J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
    d = hypot(sqrt(2) * norm(Σ - I), 0.5 * norm(A * J - J * A))
    return d
end

N = [10, 100, 1000] .÷ 2#floor.(Int, 10 .^ LinRange(log10(10), log10(1000), 10)) .÷ 2
errors_bad_qr = zeros(length(N))
errors_random_qr = zeros(length(N))
errors_nice_qr = zeros(length(N))
errors_bad_chol = zeros(length(N))
errors_random_chol = zeros(length(N))
errors_nice_chol = zeros(length(N))
errors_bad_qr_J = zeros(length(N))
errors_random_qr_J = zeros(length(N))
errors_nice_qr_J = zeros(length(N))
errors_bad_chol_J = zeros(length(N))
errors_random_chol_J = zeros(length(N))
errors_nice_chol_J = zeros(length(N))
conditionings_qr = zeros(length(N))
conditionings_chol = zeros(length(N))
Ntrials = 10
for (i, n) ∈ enumerate(N)
    for trial in 1:Ntrials
        J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
        #bad case
        V = qr(randn(2n, 2n)).Q
        Σ = exp2.(-LinRange(0, 52, n))
        S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
        A = V * S * V'
        A = (A - A') / 2
        conditionings_qr[i] += log2(cond(A)) / Ntrials
        P = polarqr!(copy(A))
        M = getM(P)
        H = M' * A * M
        errors_bad_qr[i] += log2(dist_to_Hamiltonian(H)) / Ntrials
        errors_bad_qr_J[i] += log2(dist_to_symplectic(V'M)/sqrt(n)) / Ntrials#log2(norm(P * M - M * J) / norm(P)) / Ntrials
        #errors_bad_qr_J[i] += log2(norm(J * V' * M - V' * M * J) / norm(P)) / Ntrials
        V = qr(randn(2n, 2n)).Q
        Σ = exp2.(-LinRange(0, 52, n))
        S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
        A = V * S * V'
        A = (A - A') / 2
        P = polarhybrid!(copy(A))
        M = getM(P)
        H = M' * A * M
        errors_bad_chol[i] += log2(dist_to_Hamiltonian(H)) / Ntrials
        errors_bad_chol_J[i] += log2(norm(P * M - M * J) / norm(P)) / Ntrials
        #random case
        A = randn(2n, 2n)
        A = (A - A') / 2
        P = polarqr!(copy(A))
        M = getM(P)
        H = M' * A * M
        errors_random_qr[i] += log2(dist_to_Hamiltonian(H)) / Ntrials
        errors_random_qr_J[i] += log2(norm(P * M - M * J) / norm(P)) / Ntrials
        P = polarhybrid!(copy(A))
        M = getM(P)
        H = M' * A * M
        errors_random_chol[i] += log2(dist_to_Hamiltonian(H)) / Ntrials
        errors_random_chol_J[i] += log2(norm(P * M - M * J) / norm(P)) / Ntrials

        #nice case
        V = qr(randn(2n, 2n)).Q
        Σ = 1 .+ rand(n)
        S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
        A = V * S * V'
        A = (A - A') / 2
        P = polarqr!(copy(A))
        M = getM(P)
        H = M' * A * M
        errors_nice_qr[i] += log2(dist_to_Hamiltonian(H)) / Ntrials
        errors_nice_qr_J[i] += log2(norm(J * V' * M - V' * M * J) / norm(P)) / Ntrials
        P = polarhybrid!(copy(A))
        M = getM(P)
        H = M' * A * M
        errors_nice_chol[i] += log2(dist_to_Hamiltonian(H)) / Ntrials
        errors_nice_chol_J[i] += log2(norm(P * M - M * J) / norm(P)) / Ntrials

    end
end
errors_bad_qr .= 2 .^errors_bad_qr
errors_random_qr .= 2 .^errors_random_qr
errors_nice_qr .= 2 .^errors_nice_qr
conditionings_qr .= 2 .^conditionings_qr
errors_bad_chol .= 2 .^errors_bad_chol
errors_random_chol .= 2 .^errors_random_chol
errors_nice_chol .= 2 .^errors_nice_chol
conditionings_chol .= 2 .^conditionings_chol
errors_bad_qr_J .= 2 .^errors_bad_qr_J
errors_random_qr_J .= 2 .^errors_random_qr_J
errors_nice_qr_J .= 2 .^errors_nice_qr_J
errors_bad_chol_J .= 2 .^errors_bad_chol_J
errors_random_chol_J .= 2 .^errors_random_chol_J
errors_nice_chol_J .= 2 .^errors_nice_chol_J

P = plot(framestyle=:box, xscale=:log10, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12)
plot!(P, 2N, errors_bad_qr, label=L"\kappa(A) \approx ε_\mathrm{m}^{-1}", marker=:circle, linestyle=:solid, color=:blue)
plot!(P, 2N, errors_random_qr, label=L"\kappa(A) =\mathcal{O}(n)", marker=:square, linestyle=:solid, color=:green)
plot!(P, 2N, errors_nice_qr, label=L"\kappa(A) =\mathcal{O}(1)", marker=:dtriangle, linestyle=:solid, color=:red)
#plot!(P, 2N, errors_bad_qr_J, label=L"\kappa(A) \approx ε_\mathrm{m}^{-1}", marker=:circle, linestyle=:dash, color=:blue)
#plot!(P, 2N, errors_random_qr_J, label=L"\kappa(A) =\mathcal{O}(n)", marker=:square, linestyle=:dash, color=:green)
#plot!(P, 2N, errors_nice_qr_J, label=L"\kappa(A) =\mathcal{O}(1)", marker=:dtriangle, linestyle=:dash, color=:red)
plot!(P, 2N, 2N * eps(Float64), label= L"\mathcal{O}(n \cdot \varepsilon_\mathrm{m}) ", linestyle=:dot, color =:black)
#plot!(P, 2N, sqrt.(2N) .* eps(Float64) .* 3, label= L"\mathcal{O}(\sqrt{n} \cdot\varepsilon_\mathrm{m}) ", linestyle=:dot, color =:black)
#plot!(P, 2N, conditionings_qr * eps(Float64), marker = :diamond, label= L"\kappa(A) \cdot \varepsilon_\mathrm{m} ", linestyle=:dashdot, color=:green)
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, L"\Vert A_\mathcal{H} - \mathrm{skh}(A_\mathcal{H})\ \Vert_\mathrm{F} / \Vert A\ \Vert_\mathrm{F}")
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/hamiltonian_form_accuracy_qr.pdf")
#savefig(P, path)
display(P)

P = plot(framestyle=:box, xscale=:log10, yscale=:log10, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12)
plot!(P, 2N, errors_bad_chol, label=L"\kappa(A) \approx \varepsilon_\mathrm{m}^{-1}", marker=:circle, linestyle=:solid, color=:blue)
plot!(P, 2N, errors_random_chol, label=L"\kappa(A) =\mathcal{O}(n)", marker=:square, linestyle=:solid, color=:green)
plot!(P, 2N, errors_nice_chol, label=L"\kappa(A) =\mathcal{O}(1)", marker=:dtriangle, linestyle=:solid, color=:red)
#plot!(P, 2N, errors_bad_chol_J, label=L"\kappa(A) \approx \sqrt{\varepsilon_\mathrm{m}^{-1}}", marker=:circle, linestyle=:dash, color=:blue)
#plot!(P, 2N, errors_random_chol_J, label=L"\kappa(A) =\mathcal{O}(n)", marker=:square, linestyle=:dash, color=:green)
#plot!(P, 2N, errors_nice_chol_J, label=L"\kappa(A) =\mathcal{O}(1)", marker=:dtriangle, linestyle=:dash, color=:red)
plot!(P, 2N, 2N * eps(Float64) * 3, label= L"\mathcal{O}(n \cdot \varepsilon_\mathrm{m}) ", linestyle=:dot, color =:black)
#plot!(P, 2N, sqrt.(2N) .* eps(Float64) .* 3, label= L"\mathcal{O}(\sqrt{n} \cdot\varepsilon_\mathrm{m}) ", linestyle=:dot, color =:black)
#plot!(P, 2N, conditionings_qr * eps(Float64), marker = :diamond, label= L"\kappa(A) \cdot \varepsilon_\mathrm{m} ", linestyle=:dashdot, color=:green)
xlabel!(P, "Matrix size " * L"2n")
ylabel!(P, L"\Vert A_\mathcal{H} - \mathrm{skh}(A_\mathcal{H})\ \Vert_\mathrm{F} / \Vert A\ \Vert_\mathrm{F}")
script_dir = @__DIR__
path = joinpath(script_dir, "../figures/hamiltonian_form_accuracy_chol.pdf")
#savefig(P, path)
display(P)

println("Errors for random case with QR-based polar decomposition:")
display(errors_random_chol)
println("Errors for bad case with QR-based polar decomposition:")
display(errors_bad_chol)
println("Errors for nice case with QR-based polar decomposition:")
display(errors_nice_chol)
