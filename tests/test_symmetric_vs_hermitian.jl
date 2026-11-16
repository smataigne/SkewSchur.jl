using LinearAlgebra

nthreads = 4
BLAS.set_num_threads(nthreads)
n = 1000
S = randn(2n, 2n)
S = Symmetric(S)  # Make it symmetric
H = randn(ComplexF64, n, n)
H = Hermitian(H)  # Make it Hermitian

tsym = @elapsed eigen(S)
therm = @elapsed eigen(H)
println("Time for symmetric eigenvalue decomposition: $tsym seconds")
println("Time for Hermitian eigenvalue decomposition: $therm seconds")