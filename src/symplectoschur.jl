using LinearAlgebra, SkewLinearAlgebra, BenchmarkTools
include("polar.jl")

@views function signhalley!(X::AbstractMatrix{T}) where T
    n = size(X, 1)
    ε = n * eps(T)
    X ./= norm(X)
    e = 1
    H = zeros(n, n)
    itermax = 30
    iter = 0
    temp = zeros(n, n)
    while e > ε && iter < itermax
        mul!(H, X, X, -1, 0)
        mul!(temp, X, H, 1, 0)
        X .*= 3
        X .+= temp
        H .*= 3
        H .+= I(n)
        X = X / H
        #X = X * (3I - X*X) * inv(I - 3X*X)
        e = norm(X + X') / (2 * sqrt(n))
        #display(e)
        iter += 1
    end
    skewhermitian!(X)
    return X   
end

@views function symplectoschur!(X::AbstractMatrix{T}; skew = true, polar = :qr) where T
    n  = size(X, 1) ÷ 2
    Q = zeros(2n, 2n)
    U1 = zeros(T, n, n)
    U2 = zeros(T, n, n)
    #use Q as temp to reduce allocations
    if skew
        Q .= X
        Q .-= X'
        Q .*= 0.5
        if polar == :qr 
            P = polarqr!(Q)
        elseif polar == :chol
            P = polarchol!(Q)
        elseif polar == :hybrid
            P = polarhybrid!(Q)
        else
            P = polarsvd!(Q)
        end
    else
        P = signhalley!(copy(X))
    end
    V = randn(T, 2n, n)
    temp = zeros(2n, n)
    mul!(temp, P, V, 1, 0)
    U = complex.(temp[:, 1:n], V)
    M = Matrix(qr(U).Q)
    #Use Q as temp
    temp = Q[:, 1:n]
    Mr = real.(M) .* sqrt(2)    #reduce allocations
    Mi = imag.(M) .* (-sqrt(2))
     #reduce allocations
    mul!(temp, X, Mr , 1, 0)
    mul!(U1, Mr', temp, 1, 0)
    mul!(U2, Mi', temp, 1, 0)
    H2 = complex.(U1, U2)
    E = eigen!(H2)
    U1 .= real.(E.vectors)    #reduce allocations
    U2 .= imag.(E.vectors)
    mul!(Q[:, 1:n], Mr, U1, 1, 0)
    mul!(Q[:, 1:n], Mi, U2, 1, 1)
    mul!(Q[:, n+1:2n], Mr, U2, -1, 0)
    mul!(Q[:, n+1:2n], Mi, U1, 1, 1)
    C = Diagonal(real.(E.values))
    S = Diagonal(imag.(E.values))
    return Q, C, S
end
#=
BLAS.set_num_threads(4)
X = Matrix(qr(randn(100, 100)).Q)
if det(X) < 0
    X[:,1] .= -X[:,1]
end

Q, C, S = symplectoschur!(copy(X); skew = true, polar = :hybrid)
display(norm(X * Q - Q * [C -S; S C])/ norm(X))
#=
Q, C, S = symplectoschur!(copy(X); skew = true, polar = :hybrid)
display(norm(X * Q - Q * [C -S; S C])/ norm(X))
Q, C, S = symplectoschur!(copy(X); skew = false, polar = :qr)
display(norm(X * Q - Q * [C -S; S C])/ norm(X))
=#
NT = 10
#=
@btime symplectoschur!(copy(X); skew = true, polar = :hybrid) 
@btime symplectoschur!(copy(X); skew = false, polar = :qr) 
@btime schur(X) 
@btime eigen(X) 
=#
t1 = Inf; t2=Inf; t3=Inf; t4=Inf
for _ in 1:NT
    global t1, t2, t3, t4
    t1 = min(t1, @elapsed symplectoschur!(copy(X); skew = true, polar = :hybrid))
    t2 = min(t2, @elapsed symplectoschur!(copy(X); skew = false, polar = :qr))
    t3 = min(t3, @elapsed schur(X))
    t4 = min(t4, @elapsed eigen(X))
end
display(t1)
display(t2)
display(t3)
display(t4)
println("Done")
=#
