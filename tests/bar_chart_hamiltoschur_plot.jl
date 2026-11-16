using DelimitedFiles, LinearAlgebra, SkewLinearAlgebra, Plots, PythonPlot

timings_file = joinpath(@__DIR__, "../figures/hamiltoschur_time_bar.dat")
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
fig, subplot_axes = subplots(1, 3, figsize = (11, 4))
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