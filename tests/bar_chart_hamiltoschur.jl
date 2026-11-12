using DelimitedFiles, LinearAlgebra, SkewLinearAlgebra, Plots, PythonPlot
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
BLAS.set_num_threads(4)
n = 100
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
        if type == "good"
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
labels = ["Polar factor" "QR factorization" "Matrix multiplications" "Hermitian EVD"]

# Create one stacked bar for each algorithm, with one segment per timing step.
algorithms = ["QDWH", "Newt.-Schulz", "PSVD"]

timings = Float64.(readdlm(timings_file))
timings_random = timings[:, 1:3]
timings_good = timings[:, 4:6]
timings_bad = timings[:, 7:9]
hatches = ["", "//", "\\", "xx"]
colors = ["#B3DDF2", "#C7E9C0", "#FDD0A2", "#DADAEB"]
pythonplot()
PythonPlot.matplotlib.rcParams["font.family"] = "serif"
PythonPlot.matplotlib.rcParams["font.serif"] = ["Computer Modern"]
PythonPlot.matplotlib.rcParams["mathtext.fontset"] = "cm"
PythonPlot.matplotlib.rcParams["text.usetex"] = true
PythonPlot.matplotlib.rcParams["axes.titlesize"] = 15
PythonPlot.matplotlib.rcParams["axes.labelsize"] = 15
PythonPlot.matplotlib.rcParams["xtick.labelsize"] = 15
PythonPlot.matplotlib.rcParams["ytick.labelsize"] = 15
PythonPlot.matplotlib.rcParams["legend.fontsize"] = 15
fig, subplot_axes = subplots(1, 3, figsize = (15, 4))
timings_by_matrix = [timings_random, timings_good, timings_bad]
titles = [
    "Random Gaussian matrix",
    "Random well-conditioned matrix",
    "Random ill-conditioned matrix"
]
plot_axes = [subplot_axes[i] for i in 0:2]
for (ax, timing_matrix, title) in zip(plot_axes, timings_by_matrix, titles)
    timings_by_algorithm = timing_matrix'
    bottom = zeros(length(algorithms))
    totals = vec(sum(timings_by_algorithm, dims = 2))
    bar_positions = 0:(length(algorithms) - 1)
    for j in axes(timings_by_algorithm, 2)
        values = timings_by_algorithm[:, j]
        ax.bar(
            algorithms,
            values,
            bottom = bottom,
            label = labels[j],
            color = colors[j],
            hatch = hatches[j],
            edgecolor = "black"
        )
        percentages = 100 .* values ./ totals
        centers = bottom .+ values ./ 2
        if isone(j)
            for i in eachindex(algorithms)
                ax.text(
                    bar_positions[i],
                    centers[i],
                    "$(round(percentages[i]; digits = 1))\\%",
                    ha = "center",
                    va = "center",
                    fontsize = 15
                )
            end
        end
        bottom .+= values
    end
    ax.set_title(title)
    ax.set_axisbelow(true)
    ax.grid(axis = "y", linestyle = "--", alpha = 0.7)
end
plot_axes[1].set_ylabel("Time [s]")
handles, legend_labels = plot_axes[1].get_legend_handles_labels()
fig.legend(
    handles,
    legend_labels,
    loc = "lower center",
    bbox_to_anchor = (0.5, 0.0),
    ncol = 4
)
fig.tight_layout(rect = (0, 0.12, 1, 1))
fig.savefig(joinpath(@__DIR__, "../figures/hamiltoschur_time_bar.pdf"))

println("done")
