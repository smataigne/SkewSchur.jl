using LinearAlgebra

BLAS.set_num_threads(32)
n = 4000
X = randn(2n, 2n)
temp1 = zeros(2n, 2n)
temp2 = zeros(2n, 2n)
t1 = @elapsed BLAS.syrk!('U', 'T', 1.0, X, 0.0, temp1)
t2 = @elapsed mul!(temp2, X', X, 1.0, 0.0)
display("Time for BLAS.syrk: $t1")
display("Time for mul!: $t2")
display(norm(Hermitian(temp1) - Hermitian(temp2)))