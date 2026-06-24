using LinearAlgebra

Ω = randn(4, 4)
Ω = Ω - Ω'  # Make Ω skew-symmetric

theta = atan(Ω[2,1] + Ω[3,4], Ω[3,1] - Ω[2,4]) 
G = Matrix(1.0I, 4, 4)
G[2,2] = cos(theta)
G[2,3] = -sin(theta)
G[3,2] = sin(theta)
G[3,3] = cos(theta)
Ω2 = G * Ω * G'
theta2 = atan(-Ω2[4,1] + Ω2[3,2], Ω2[3,1] + Ω2[4,2])
G = Matrix(1.0I, 4, 4)
G[3,3] = cos(theta2)
G[3,4] = -sin(theta2)
G[4,3] = sin(theta2)
G[4,4] = cos(theta2)
Ω2 = G * Ω2 * G'
display(Ω)
display(round.(Ω2; sigdigits=2))