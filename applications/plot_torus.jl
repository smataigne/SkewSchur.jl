using GLMakie, FileIO

img = load("./figures/wave2d.png")

R, r = 2.0, 0.7
u = range(0, 2π, length=size(img,1))
v = range(0, 2π, length=size(img,2))

x = [(R + r*cos(vv))*cos(uu) for uu in u, vv in v]
y = [(R + r*cos(vv))*sin(uu) for uu in u, vv in v]
z = [r*sin(vv) for uu in u, vv in v]

fig = Figure(size = (600, 600))
ax = Axis3(fig[1,1], aspect=:data)
surface!(ax, x, y, z, color=img)

hidedecorations!(ax)
hidespines!(ax)

lims = R + r
limits!(ax, -lims, lims, -lims, lims, -r, r)

ax.azimuth = 0.3π
ax.elevation = 0.15π

save("./figures/torus_view.png", fig, px_per_unit = 4)