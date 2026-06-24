using LinearAlgebra, Plots

@views function mgs!(A::AbstractMatrix)
    m, n = size(A)
    for j in 1:n
        for i in 1:j-1
            A[:, j] -= (A[:, i]' * A[:, j]) * A[:, i]
        end
        A[:, j] /= norm(A[:, j])
    end
    return A
end

@views function cgs!(A::AbstractMatrix)
    m, n = size(A)
    for j in 1:n
        v = copy(A[:, j])
        for i in 1:j-1
            A[:, j] -= (A[:, i]' * v) * A[:, i]
        end
        A[:, j] ./= norm(A[:, j])
    end
    return A
end

ε =   sqrt(eps(Float64))
A = [1/ε  1  1 0;
     ε  0  0 1;
     0  0  ε 0;
     0  0  0 ε]
A = [sqrt(ε) 0    0       0;
     1       1    1       1;
     0       0  sqrt(ε)   0;
     0       0    ε       sqrt(ε)]
     
A = [1 1 1;
     ε  0  0;
     0  0  ε]
     #=
A = [1 1;
     ε 0]=#
#=
A = [1 1;
     √ε 0]=#
#A = [1 1 1;
#     sqrt(ε)  0  0;
#     0  0  sqrt(ε)]
#=
e = sqrt(ε)
A = [1 1 1 1;
     e  0  0 e;
     0  0  e e;
     0  0  0 e]
     =#
e= 0.01
A = [1 1 1;
    e  e  0;
     e^2  0  0]
Q_mgs = mgs!(copy(A))
Q_cgs = cgs!(copy(A))
display(norm(Q_mgs' * Q_mgs - I))
display(norm(Q_cgs' * Q_cgs - I))
display(cond(A))

E = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
lcgs = zeros(length(E))
lmgs = zeros(length(E))
cA = zeros(length(E))
for (i, e) in enumerate(E)
    A = BigFloat.([1 1 1;
         e  e  0;
        e  0  0])
    Q_mgs = mgs!(copy(A))
    Q_cgs = cgs!(copy(A))
    lcgs[i] = norm(Q_cgs' * Q_cgs - I)
    lmgs[i] = norm(Q_mgs' * Q_mgs - I)
    cA[i] = cond(Float64.(A))
end
P = plot()
plot!(cA, [lcgs lmgs], labels=["CGS" "MGS"], xaxis=:log, yaxis=:log)
plot!(cA,  cA.^2 .* eps(BigFloat) , label="cond(A)^2", xaxis=:log, yaxis=:log)
plot!(cA, cA.^3 .* eps(BigFloat), label="cond(A)^3", xaxis=:log, yaxis=:log)
display(P)