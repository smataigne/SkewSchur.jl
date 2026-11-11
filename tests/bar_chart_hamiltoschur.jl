using LinearAlgebra, SkewLinearAlgebra, Plots
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
        elseif polar == :newton
            P = polarnewtonschulz!(copy(A))
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

tchol = zeros(4)
tsvd = zeros(4)
thybrid = zeros(4)
Ntrials = 2
for i in 1:Ntrials
    A = randn(1000, 1000)
    A = (A - A')/2
    tchol .+= time_hamiltoschur!(copy(A), :chol) / Ntrials
    tsvd .+= time_hamiltoschur!(copy(A), :svd) / Ntrials
    thybrid .+= time_hamiltoschur!(copy(A), :hybrid) / Ntrials
end



labels = ["Polar factor" "QR factorization" "Matrix multiplications" "Hermitian EVD"]

# Create one stacked bar for each algorithm, with one segment per timing step.
algorithms = ["QDWH", "CDWH", "PSVD"]
timings = [thybrid tchol tsvd]
pythonplot()
P = bar(
    algorithms,
    timings',
    label = labels,
    title = "Polar decomposition methods",
    size = (700, 400),
    ticksfontfamily    = "Computer Modern",
    fontfamily         = "Computer Modern",
    legendfontfamily   = "Computer Modern",
    titlefontfamily    = "Computer Modern",
    legendfontsize  = 15,
    labelfontsize   = 15,
    tickfontsize    = 10,
    titlefontsize   = 15,
    margin = 2Plots.mm,
    ylabel = "Time [s]",
    legend = :outerright
)

savefig(P, joinpath(@__DIR__, "../figures/hamiltoschur_time_bar.pdf"))

println("done")
