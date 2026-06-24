using LinearAlgebra, BenchmarkTools, SkewLinearAlgebra

#QDWH algroithm for polar decomposition
#Nakatsukasa, Bai, Gygi, "Optimizing Halley's iteration for computing the matrix polar decomposition", SIAM J. Matrix Anal. Appl., 2010.

function h(l::Number)
    d = cbrt(4 * (1 - l * l)/ l^4)
    c = sqrt(1 + d)
    return c + 0.5 * sqrt(8 - 4 * d + 8 * (2 - l * l) / (l * l * c))
end

@views function qdwh_qr!(A::AbstractMatrix{T}) where T
    
    n = size(A, 1)
    ε = n * eps(T)
    α = norm(A)
    β = eps(T)
    try 
        β = 1 / (opnorm(inv(A), 1) * sqrt(n))
    catch
        β = eps(T)
    end
    X = A ./ α
    l = β / α
    temp = zeros(2n, n)
    Q = zeros(2n, n)
    H = zeros(n, n)
    e = 1
    while e > ε
        a = h(l)
        b = (a - 1)^2 / 4
        c = a + b - 1
        temp[1:n, 1:n] .= sqrt(c) * X
        temp[n+1:2n, 1:n] .= I(n)
        Q .= Matrix(qr(temp).Q)
        mul!(X, Q[1:n, 1:n], Q[n+1:2n, 1:n]', (a - b/c)/sqrt(c), b / c)
        l = l * (a + b *l *l)/(1 + c * l * l)
        mul!(H, X', X, 1, 0)
        e = norm(H - I(n))
        display(e)
    end 
    return X
end

@views function qdwh_sol!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    ε = n * eps(T)
    
    α = norm(A)
    β = 1 / (opnorm(inv(A), 1) * sqrt(n))
    X = A ./ α
    l = β / α
    temp = zeros(n, n)
    H = zeros(n, n)
    mul!(H, X', X, 1, 0)
    e = 1
    while e > ε
        a = h(l)
        b = (a - 1)^2 / 4
        c = a + b - 1
        mul!(temp, X, Symmetric(H), 1, 0)
        X .*= a
        @. X += b * temp
        H .*= c
        H .+= I(n)
        X = X / Symmetric(H) 
        l = l * (a + b *l *l)/(1 + c * l * l)
        mul!(H, X', X, 1, 0)
        e = norm(H - I(n))
        #display(e)
    end 
    return X
end

@views function qdwh_hybrid!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    ε = n * eps(T)
    
    α = norm(A)
    β = 1 / (opnorm(inv(A), 1) * sqrt(n))
    X = A ./ α
    l = β / α
    temp = zeros(n, n)
    H = zeros(n, n)
    mul!(H, X', X, 1, 0)
    e = norm(H - I(n))
    iter = 0
    #Perform globally convergent Nakatsukasa steps
    while e > 0.01 * sqrt(n)
        iter += 1
        a = h(l)
        b = (a - 1)^2 / 4
        c = a + b - 1
        mul!(temp, X, Symmetric(H), 1, 0)
        X .*= a
        @. X += b * temp
        H .*= c
        H .+= I(n)
        X = X / Symmetric(H) 
        l = l * (a + b *l *l)/(1 + c * l * l)
        mul!(H, X', X, 1, 0)
        e = norm(H - I(n))
        #display(e)
    end
    #Local convergence with BLAS3 Newton-Schulz iteration
    while e > ε
        mul!(temp, X, Symmetric(H), 1, 0)
        X .*= 3/2
        X .-= temp/2
        skewhermitian!(X)
        mul!(H, X', X, 1, 0)
        e = norm(H - I(n))
        #display(e)
    end 
    return X
end

#Halley's iteration for polar decomposition
@views function halley!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    ε = n * eps(T)
    X = A / norm(A)
    e = 1
    H = zeros(n, n)
    mul!(H, X', X)
    temp = zeros(n, n)
    itermax = 30
    while e > ε && itermax > 0
        itermax -= 1
        mul!(temp, X, Symmetric(H), 1, 0)
        X .*= 3
        X .+= temp
        H .*= 3
        H .+= I(n)
        X = X / Symmetric(H)
        mul!(H, X', X, 1, 0)
        e = norm(H - I(n))
        display(e)
    end
    return X    

end
#= Test speed of the three algorithms =#
BLAS.set_num_threads(4)
n = 4
nodd = 2 * n + 1
A = randn(nodd, nodd)
A = A - A'  # Make A skew-symmetric
P = qdwh_qr!(copy(A))
#@btime qdwh_sol!(copy(A))
#@btime qdwh_hybrid!(copy(A))
#X = halley!(copy(A))
#@btime opnorm(inv(copy(A)), 1)
#@btime norm(inv(copy(A)))
#V = qdwh_qr!(copy(A))
#W = qdwh_sol!(copy(A))
x = randn(nodd)
z = x + P * (P * x)
z ./= norm(z)
V = randn(nodd, n)
V .-= z * (z' * V)
temp = zeros(nodd, n)
mul!(temp, P, V, 1, 0)
U = complex.(temp[:, 1:n], V)
M = Matrix(qr(U).Q)
#Use Q as temp
temp = Q[:, 1:n]
Mr = √2 .* real.(M)        #reduce allocations
Mi = -√2 .* imag.(M)
Z = [Mr Mi z]
display(round.(Z' * P *Z; digits=2))
display(round.(Z' * A * Z; digits=2))
println("Done!")

