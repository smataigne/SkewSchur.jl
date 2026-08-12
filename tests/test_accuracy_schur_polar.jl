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

@views function getZ(P::AbstractMatrix{T}, A::AbstractMatrix{T}; method =:qr) where T
    n = size(P, 1) ÷ 2
    V =  randn(2n,n)#randn(Complex{T}, 2n, n)
    temp = zeros(2n, 2n)
    mul!(temp[:, 1:n], P, V, 1, 0)
    #U = (P + 1im * I(2n)) * V #reduce allocations
    U = complex.(temp[:, 1:n], V) #reduce allocations
    #M = Matrix(qr!(U, NoPivot()).Q)
    #M = modified_gramschmidt!(U)
    if method == :qr
        M = Matrix(qr!(U, NoPivot()).Q)
    elseif method == :gramschmidt
        M = modified_gramschmidt!(U)
    elseif method == :polar
        M = polarhalley2(U)
    else
        V = zeros(2n, n)
        V[:, 1] = randn(2n)
        V[:, 1] ./= norm(V[:, 1])
        for j ∈ 2:n
            V[:, j] = A * A * V[:, j-1]
            for k ∈ 1:j-1
                V[:, j] .-= (V[:, k]' * V[:, j]) * V[:, k]
            end
            V[:, j] ./= norm(V[:, j])
        end
    end
    #println("norm(M'M - I) / sqrt(n) = $(norm(M'M - I) / sqrt(n))")
    #println("Equivalent to : $(norm(2 * real(M)' * real(M) + 2 * imag(M)' * imag(M) + 2im * (real(M)' * imag(M) - imag(M)' * real(M)) - 2I) / sqrt(4n))")
    #println("See also: $(norm(2 * real(M)' * real(M) + 2 * imag(M)' * imag(M)  - 2I) / sqrt(4n))")
    temp[:, 1:n] .=  √2 *real(M)     #reduce allocations
    temp[:, n+1:2n] .=-√2 * imag(M)
    #classical_gramschmidt!(temp)
    #=
    QR1 = qr(√2 * real(M))       #reduce allocations
    temp[:, 1:n] .= Matrix(QR1.Q) * Diagonal(diag(QR1.R))      
    QR2 = qr((I - temp[:, 1:n] * temp[:, 1:n]') * √2 * imag(M))
    #Orthogonalize G with respect to temp[:, 1:n]
    temp[:, n+1:2n] .= -Matrix(QR2.Q) * Diagonal(diag(QR2.R)) #-Matrix(qr((I - temp[:, 1:n] * temp[:, 1:n]') * G).Q)
    =#
    #=
    QR1 = qr(√2 * real(M))       #reduce allocations
    temp[:, 1:n] .= Matrix(QR1.Q) * Diagonal(sign.(diag(QR1.R)))
    QR2 = qr((I - temp[:, 1:n] * temp[:, 1:n]') * (-√2 * imag(M)))       #reduce allocations
    temp[:, n+1:2n] .= Matrix(QR2.Q) * Diagonal(sign.(diag(QR2.R))) #-Matrix(qr((I - temp[:, 1:n] * temp[:, 1:n]') * G).Q)
    =#
    #temp[:, n+1:2n] .= (-√2 * imag(M) / QR1.R)  *  Diagonal(sign.(diag(QR1.R)))#-Matrix(qr((I - temp[:, 1:n] * temp[:, 1:n]') * G).Q)
    #temp[:, 1:n] = classical_gramschmidt!(√2 * real(M))        #reduce allocations
    #temp[:, n+1:2n] = classical_gramschmidt!(-√2 * imag(M))
    #println("norm(temp' * temp - I) / sqrt(2n) = $(norm(temp' * temp - I) / sqrt(2n))")
    #display(norm(temp[:, 1:n]' * temp[:, n+1:2n]) / sqrt(n))
    #display(norm(temp[:, 1:n]' * temp[:, 1:n] - I) / sqrt(n))
    #display(norm(temp[:, n+1:2n]' * temp[:, n+1:2n] - I) / sqrt(n))
    #QR = qr(temp)
    #temp .= Matrix(QR.Q) * Diagonal(sign.(diag(QR.R)))
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

Ns = floor.(Int, 2 .^ LinRange(log2(16), log2(4096), 5)) .÷ 2
Ntrials = 5
errors_getz_qr = zeros(length(Ns))
errors_getz_gramschmidt = zeros(length(Ns))
errors_getz_polar = zeros(length(Ns))
errors_schur = zeros(length(Ns))
errors_orthogonality = zeros(length(Ns))
for (i, n) in enumerate(Ns)
    for trial in 1:Ntrials
        Q = Matrix(qr(randn(2n, 2n)).Q)
        J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
        v = rand(n)
        S = [zeros(n, n) -Diagonal(v); Diagonal(v) zeros(n, n)]
        P = Q * J * Q'
        A = Q * S * Q'
        P = (P - P') / 2
        Zqr = getZ(P, A, method = :qr)
        #Zgramschmidt = getZ(P, A, method = :gramschmidt)
        #Zpolar = getZ(P, A, method = :polar)
        
        
        #=
        M = Q'Z
        display(norm(M'M - I) / sqrt(n))
        display(norm(J * M - M * J) / sqrt(n))
        display(dist_to_symplectic(M)/ sqrt(n))
        display(norm(P * Z - Z * J) / norm(P))
        =#
        errors_getz_qr[i] += log2(norm(P * Zqr - Zqr * J) / norm(P)) / Ntrials
        #errors_getz_gramschmidt[i] += log2(norm(P * Zgramschmidt - Zgramschmidt * J) / norm(P)) / Ntrials
        #errors_getz_polar[i] += log2(norm(P * Zpolar - Zpolar * J) / norm(P)) / Ntrials
        errors_orthogonality[i] += log2(norm(Zqr'Zqr - I) / sqrt(2n)) / Ntrials

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

errors_getz_qr .= 2 .^errors_getz_qr
#errors_getz_gramschmidt .= 2 .^errors_getz_gramschmidt
#errors_getz_polar .= 2 .^errors_getz_polar
errors_schur .= 2 .^errors_schur
errors_orthogonality .= 2 .^errors_orthogonality


P = plot(framestyle=:box, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft, xticks = [16, 64, 256, 1024, 4096])
plot!(2Ns, errors_getz_qr, label = "RSD with steps 1 to 4", xscale = :log2, yscale = :log10, marker=:circle, color=:red)
#plot!(2Ns, errors_getz_gramschmidt, label = "RSD with Gram-Schmidt", xscale = :log10, yscale = :log10, marker=:diamond, color=:green)
#plot!(2Ns, errors_getz_polar, label = "RSD with Polar Decomposition", xscale = :log10, yscale = :log10, marker=:square, color=:blue)
#plot!(2Ns, errors_orthogonality, label = L"\Vert Z^\top Z - I\ \Vert_\mathrm{F} / \sqrt{2n}", xscale = :log10, yscale = :log10, marker=:diamond, color=:green)
plot!(2Ns, errors_schur, label = "RSD with " * L"\texttt{schur}\ (\texttt{gees})",  marker=:square, color=:blue)
#plot!(2Ns, 2Ns .* eps(Float64) .* 3, label = L"\mathcal{O}(\epsilon_\mathrm{m} \cdot n)", linestyle=:dash, color=:black)
plot!(2Ns, 2Ns .* eps(Float64) .* 2, label = L"\mathcal{O}(\epsilon_\mathrm{m} \cdot n)", linestyle=:dash, color=:black)
xlabel!("Matrix size " * L"2n")
ylabel!(L"\Vert PZ - Z J\ \Vert_\mathrm{F} / \sqrt{2n}")
display(P)
savefig(P, joinpath(@__DIR__, "../figures/getZ_accuracy.pdf"))
