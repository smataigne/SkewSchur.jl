using LinearAlgebra, Plots, LaTeXStrings
include("../src/polar.jl")

@views function classical_gramschmidt!(A::AbstractMatrix)
    n = size(A, 2)
    for j in 1:n
        for i in 1:(j-1)
            A[:, j] .-= (A[:, i]' * A[:, j]) * A[:, i]
        end
        A[:, j] ./= norm(A[:, j])
    end
    return A
end

@views function modified_gramschmidt!(A::AbstractMatrix)
    n = size(A, 2)
    for i in 1:n
        A[:, i] ./= norm(A[:, i])
        for j in (i+1):n
            A[:, j] .-= (A[:, i]' * A[:, j]) * A[:, i]
        end
    end
    return A
end

@views function polarhalley2(A::AbstractMatrix)
    n = size(A, 2)
    A ./= norm(A)
    iter = 20
    e = Inf
    while e > 1e-15 && iter > 0
        A = A * (3 * I(n) + A'A) / Hermitian(I(n) + 3 * A'A) #Halley
        iter -= 1
        e = norm(A'A - I(size(A, 2))) / norm(A)
    end
    return A
end

@views function getZ(P::AbstractMatrix{T}, A::AbstractMatrix{T}; subspace_iter = 1) where T
    n = size(P, 1) ÷ 2
    V =  randn(2n,n)#randn(Complex{T}, 2n, n)
    temp = zeros(2n, 2n)
    mul!(temp[:, 1:n], P, V, 1, 0)
    U = Matrix(qr!(complex.(temp[:, 1:n], V), NoPivot()).Q)
    if  subspace_iter > 1
        M = Matrix(qr!((P + 1im * I(2n)) * U, NoPivot()).Q)
    else
        M = U
    end
    
    temp[:, 1:n] .=  √2 *real(M)     #reduce allocations
    temp[:, n+1:2n] .=-√2 * imag(M)
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
BLAS.set_num_threads(16)
Ns = floor.(Int, 2 .^ LinRange(log2(16), log2(4096), 5)) .÷ 2
Ntrials = 10
errors_getz_qr_1 = zeros(length(Ns))
errors_getz_qr_2 = zeros(length(Ns))
errors_schur = zeros(length(Ns))
errors_orthogonality_qr_1 = zeros(length(Ns))
errors_orthogonality_qr_2 = zeros(length(Ns))
for (i, n) in enumerate(Ns)
    for trial in 1:Ntrials
        Q = Matrix(qr(randn(2n, 2n)).Q)
        J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
        v = rand(n)
        S = [zeros(n, n) -Diagonal(v); Diagonal(v) zeros(n, n)]
        P = Q * J * Q'
        A = Q * S * Q'
        P = (P - P') / 2
        Zqr1 = getZ(P, A; subspace_iter = 1)
        errors_getz_qr_1[i] += log2(norm(P * Zqr1 - Zqr1 * J) / norm(P)) / Ntrials
        Zqr2 = getZ(P, A; subspace_iter = 2)
        errors_getz_qr_2[i] += log2(norm(P * Zqr2 - Zqr2 * J) / norm(P)) / Ntrials
        errors_orthogonality_qr_1[i] += log2(norm(Zqr1'Zqr1 - I) / sqrt(2n)) / Ntrials
        errors_orthogonality_qr_2[i] += log2(norm(Zqr2'Zqr2 - I) / sqrt(2n)) / Ntrials
        Schur = schur(P)
        V = Schur.Z[:,[1:2:end; 2:2:end]]
        d = zeros(n)
        for i ∈ 1:2:(2n-1)
            d[(i÷2) + 1] = sign(Schur.T[i+1, i])
        end
        S = [zeros(n, n) -Diagonal(d); Diagonal(d) zeros(n, n)]
        errors_schur[i] += log2(norm(P * V - V * S)/ norm(P)) / Ntrials
    end
end

errors_getz_qr_1 .= 2 .^errors_getz_qr_1
errors_getz_qr_2 .= 2 .^errors_getz_qr_2
errors_schur .= 2 .^errors_schur
errors_orthogonality_qr_1 .= 2 .^errors_orthogonality_qr_1
errors_orthogonality_qr_2 .= 2 .^errors_orthogonality_qr_2


P = plot(framestyle=:box, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft, xticks = [16, 64, 256, 1024, 4096], yticks = [1e-15, 1e-13, 1e-11])
plot!(2Ns, errors_getz_qr_1, label = "RSD with 1 subspace iteration", xscale = :log2, yscale = :log10, marker=:circle, color=:red, linestyle=:solid)
plot!(2Ns, errors_getz_qr_2, label = "RSD with 2 subspace iterations", xscale = :log2, yscale = :log10, marker=:diamond, color=:green, linestyle=:dash)
plot!(2Ns, errors_schur, label = "RSD with " * L"\texttt{schur}\ (\texttt{gees})",  marker=:square, color=:blue, linestyle=:dashdot)
plot!(2Ns, 2Ns .* eps(Float64) .* 2, label = L"\mathcal{O}(\epsilon_\mathrm{m} \cdot n)", linestyle=:dot, color=:black)
xlabel!("Matrix size " * L"2n")
ylabel!(L"\Vert PZ - Z J\ \Vert_\mathrm{F} / \sqrt{2n}")
ylims!(P,  0.3*1e-15, 1e-11)
display(P)
savefig(P, joinpath(@__DIR__, "../figures/getZ_accuracy.pdf"))
