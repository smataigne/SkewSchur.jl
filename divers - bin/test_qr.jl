using LinearAlgebra, Plots, BenchmarkTools, LaTeXStrings

n = 800
p = 50
@views function takereal(A::AbstractMatrix)
    return real.(A)
end
A = randn(n, p)
B = randn(n, p)
H = complex.(A, B)
Q, R = qr(A)
#@btime Matrix(Q)
@btime complex.(A, B)
@btime R = takereal(H)
print("Done\n")