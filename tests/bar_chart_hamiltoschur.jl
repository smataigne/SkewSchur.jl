using DelimitedFiles, LinearAlgebra, SkewLinearAlgebra
include("../src/polar.jl")

"""
    time_hamiltoschur!(A::AbstractMatrix{T}, polar = :qr) where T

Time the Schur decomposition of a skew-symmetric matrix `A` using the polar decomposition method specified by `polar`.
"""
@views function time_hamiltoschur!(A::AbstractMatrix{T}, polar = :qr) where T
    n  = size(A, 1) ÷ 2
    Q = zeros(2n, 2n)
    H = zeros(T, n, n)
    Ω = zeros(T, n, n)
    t1 = @elapsed begin
        if polar == :qr
            P = polarqr!(copy(A))
        elseif polar == :chol
            P = polarchol!(copy(A))
        elseif polar == :hybrid
            P = polarhybrid!(copy(A))
        elseif polar == :newton || polar == :newtonsafe
            if polar == :newton
                P = polarnewtonschulz!(copy(A))
            else
                P = polarnewtonschulz!(copy(A); safe = true)
            end
        else
            P = polarsvd!(copy(A))
        end
    end
    t2 = @elapsed begin
        V = randn(T, 2n, n)
        temp = zeros(2n, n)
        mul!(temp, P, V, 1, 0)
        U = complex.(temp[:, 1:n], V)
        M = Matrix(qr(U).Q)
        #Use Q as temp
        temp = Q[:, 1:n]
        Mr = real.(M)        #reduce allocations
        Mi = imag.(M)
    end
     #reduce allocations
    t3 = @elapsed begin
        mul!(temp, A, Mr , sqrt(2), 0)
        mul!(Ω, Mr', temp, sqrt(2), 0)
        mul!(H, Mi', temp, sqrt(2), 0)
    end
    t4 = @elapsed begin
        H2 = complex.(H, Ω)
        hermitianpart!(H)
        E = eigen!(Hermitian(H2))
        H .= real.(E.vectors)    #reduce allocations
        Ω .= imag.(E.vectors)
    end
    t5 = @elapsed begin
        mul!(Q[:, 1:n], Mr, Ω, sqrt(2), 0)
        mul!(Q[:, 1:n], Mi, H, sqrt(2), 1)
        mul!(Q[:, n+1:2n], Mr, H, -sqrt(2), 0)
        mul!(Q[:, n+1:2n], Mi, Ω, sqrt(2), 1)
        for k ∈ 1:n
            Q[:, n + k] .*= sign(E.values[k])
        end
    end
    return [t1, t2, t3+t5, t4]
end
BLAS.set_num_threads(32)
n = 1000
tnewton= zeros(4, 3)
tsvd = zeros(4, 3)
thybrid = zeros(4, 3)
Ntrials = 2
types =["random", "good", "bad"]
for (j, type) in enumerate(types)
    for i in 1:Ntrials
        if type == "random"
            A = randn(2n, 2n)
            A = (A - A') / 2
        elseif type == "bad"
            V = qr(randn(2n, 2n)).Q
            Σ = exp2.(-LinRange(0, 52, n))
            S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
            A = V * S * V'
            A = (A - A') / 2
        elseif type == "good"
            V = qr(randn(2n, 2n)).Q
            Σ = 0.5 .+ rand(n)
            S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
            A = V * S * V'
            A = (A - A') / 2
        end
        if type == "good" || type == "random"
            tnewton[:, j] .+= time_hamiltoschur!(copy(A), :newton) / Ntrials
        else
            tnewton[:, j] .+= time_hamiltoschur!(copy(A), :newtonsafe) / Ntrials
        end
        tsvd[:, j] .+= time_hamiltoschur!(copy(A), :svd) / Ntrials
        thybrid[:, j] .+= time_hamiltoschur!(copy(A), :hybrid) / Ntrials
    end
end
timings_random = [thybrid[:, 1] tnewton[:, 1] tsvd[:, 1]]
timings_good = [thybrid[:, 2] tnewton[:, 2] tsvd[:, 2]]
timings_bad = [thybrid[:, 3] tnewton[:, 3] tsvd[:, 3]]

timings_file = joinpath(@__DIR__, "../figures/hamiltoschur_time_bar.dat")
writedlm(timings_file, [timings_random timings_good timings_bad])
