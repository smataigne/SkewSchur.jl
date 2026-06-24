using LinearAlgebra, Plots, LaTeXStrings
include("polar.jl")

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

@views function getZ(P::AbstractMatrix{T}) where T
    n = size(P, 1) ÷ 2
    V = randn(T, 2n, n)
    temp = zeros(T, 2n, 2n)
    mul!(temp[:, 1:n], P, V, 1, 0)
    U = complex.(temp[:, 1:n], V) #reduce allocations
    M = Matrix(qr(U).Q)
    
    #println("norm(M'M - I) / sqrt(n) = $(norm(M'M - I) / sqrt(n))")
    #println("Equivalent to : $(norm(2 * real(M)' * real(M) + 2 * imag(M)' * imag(M) + 2im * (real(M)' * imag(M) - imag(M)' * real(M)) - 2I) / sqrt(4n))")
    #println("See also: $(norm(2 * real(M)' * real(M) + 2 * imag(M)' * imag(M)  - 2I) / sqrt(4n))")
    temp[:, 1:n] .=  √2 *real(M)     #reduce allocations
    temp[:, n+1:2n] .=-√2 * imag(M)
    classical_gramschmidt!(temp)
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
#=
Ns = floor.(Int, 10 .^ LinRange(log10(10), log10(1000), 5)) .÷ 2
Ntrials = 10
errors_getz = zeros(length(Ns))
errors_schur = zeros(length(Ns))
errors_orthogonality = zeros(length(Ns))
for (i, n) in enumerate(Ns)
    for trial in 1:Ntrials
        Q = Matrix(qr(randn(2n, 2n)).Q)
        J = [zeros(n, n) -I(n); I(n) zeros(n, n)]
        P = Q * J * Q'
        P = (P - P') / 2
        Z = getZ(P)
        M = Q'Z
        #=
        display(norm(M'M - I) / sqrt(n))
        display(norm(J * M - M * J) / sqrt(n))
        display(dist_to_symplectic(M)/ sqrt(n))
        display(norm(P * Z - Z * J) / norm(P))
        =#
        errors_getz[i] += log2(norm(P * Z - Z * J) / norm(P)) / Ntrials
        errors_orthogonality[i] += log2(norm(Z'Z - I) / sqrt(2n)) / Ntrials

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

errors_getz .= 2 .^errors_getz
errors_schur .= 2 .^errors_schur
errors_orthogonality .= 2 .^errors_orthogonality
=#

P = plot(framestyle=:box, ticksfontfamily = "Computer Modern", fontfamily = "Computer Modern", legendfontfamily = "Computer Modern", titlefontfamily = "Computer Modern",
legendfontsize = 12, labelfontsize = 15, tickfontsize = 12, legend=:topleft)
plot!(2Ns, errors_getz, label = "RSD with steps 1 to 4", xscale = :log10, yscale = :log10, marker=:circle, color=:red)
#plot!(2Ns, errors_orthogonality, label = L"\Vert Z^\top Z - I\ \Vert_\mathrm{F} / \sqrt{2n}", xscale = :log10, yscale = :log10, marker=:diamond, color=:green)
plot!(2Ns, errors_schur, label = "RSD with " * L"\texttt{schur}\ (\texttt{gees})",  marker=:square, color=:blue)
#plot!(2Ns, 2Ns .* eps(Float64) .* 3, label = L"\mathcal{O}(\epsilon_\mathrm{m} \cdot n)", linestyle=:dash, color=:black)
plot!(2Ns, sqrt.(2Ns) .* eps(Float64) .* 10, label = L"\mathcal{O}(\epsilon_\mathrm{m} \cdot \sqrt{n})", linestyle=:dash, color=:black)
xlabel!("Matrix size " * L"2n")
ylabel!(L"\Vert PZ - Z J\ \Vert_\mathrm{F} / \sqrt{2n}")
display(P)
savefig(P, joinpath(@__DIR__, "./figures/getZ_accuracy.pdf"))
