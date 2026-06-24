using LinearAlgebra, Plots, BenchmarkTools, LaTeXStrings, SkewLinearAlgebra
include("polar.jl")

BLAS.set_num_threads(4)

n = 1000
Ndata = 4

lpolarchol = zeros(Ndata)
lpolarqr = zeros(Ndata)
lpolarsvd = zeros(Ndata)
lpolarhalley = zeros(Ndata)
lpolarnewton = zeros(Ndata)
lpolarhybrid = zeros(Ndata)

#Case 1 : random matrix
A = randn(2n, 2n)
A1 = (A - A') / 2

#Case 2 : well-conditioned matrix
V = qr(randn(2n, 2n)).Q
Σ = 1 .+ rand(n)
S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
A = V * S * V'
A2 = (A - A') / 2

#Case 3 : ill-conditioned matrix
V = qr(randn(2n, 2n)).Q
Σ = exp2.(-LinRange(0, 53, n))
S = [zeros(n, n) -Diagonal(Σ); Diagonal(Σ) zeros(n, n)]
A = V * S * V'
A3 = (A - A') / 2

for A ∈ [A1,A2,A3]
    global lpolarchol, lpolarqr, lpolarsvd, lpolarhalley, lpolarnewton, lpolarhybrid
    P = polarsvd!(copy(A))
    lpolarsvd[1] = norm(P'P - I(2n)) / sqrt(2n)
    lpolarsvd[2] = norm(P + P')/ (2 * norm(P))
    H = P'A
    lpolarsvd[3] = norm(H - H') / (2 * norm(H))
    P = polarhalley!(copy(A))
    lpolarhalley[1] = norm(P'P - I(2n)) / sqrt(2n)
    lpolarhalley[2] = norm(P + P')/ (2 * norm(P))
    H = P'A
    lpolarhalley[3] = norm(H - H') / (2 * norm(H))
    P = polarqr!(copy(A))
    lpolarqr[1] = norm(P'P - I(2n)) / sqrt(2n)
    lpolarqr[2] = norm(P + P')/ (2 * norm(P))
    H = P'A
    lpolarqr[3] = norm(H - H') / (2 * norm(H))
    P = polarchol!(copy(A))
    lpolarchol[1] = norm(P'P - I(2n)) / sqrt(2n)
    lpolarchol[2] = norm(P + P')/ (2 * norm(P))
    H = P'A
    lpolarchol[3] = norm(H - H') / (2 * norm(H))
    P = polarnewtonschulz!(copy(A))
    lpolarnewton[1] = norm(P'P - I(2n)) / sqrt(2n)
    lpolarnewton[2] = norm(P + P')/ (2 * norm(P))
    H = P'A
    lpolarnewton[3] = norm(H - H') / (2 * norm(H))
    P = polarhybrid!(copy(A))
    lpolarhybrid[1] = norm(P'P - I(2n)) / sqrt(2n)
    lpolarhybrid[2] = norm(P + P')/ (2 * norm(P))
    H = P'A
    lpolarhybrid[3] = norm(H - H') / (2 * norm(H))

    lpolarsvd[4] = @belapsed polarsvd!(copy($A))
    lpolarqr[4] = @belapsed polarqr!(copy($A))
    lpolarchol[4] = @belapsed polarchol!(copy($A))
    lpolarhybrid[4] = @belapsed polarhybrid!(copy($A))
    lpolarhalley[4] = @belapsed polarhalley!(copy($A))
    lpolarnewton[4] = @belapsed polarnewtonschulz!(copy($A))

    lpolarsvd = round.(lpolarsvd, sigdigits = 2)
    lpolarqr = round.(lpolarqr, sigdigits = 2)
    lpolarchol = round.(lpolarchol, sigdigits = 2)
    lpolarhalley = round.(lpolarhalley, sigdigits = 2)
    lpolarnewton = round.(lpolarnewton, sigdigits = 2)
    lpolarhybrid = round.(lpolarhybrid, sigdigits = 2)

    println("\\hline")
    print("PSVD & ", lpolarsvd[1], " & ", lpolarsvd[3], " & ", lpolarsvd[4], "\\\\ \n")
    print("NS~\\cite{BjorckBowie1971} & ", lpolarnewton[1], " & ", lpolarnewton[3], " & ", lpolarnewton[4], "\\\\ \n")
    print("Halley~\\cite{Gander1990Polar}  & ", lpolarhalley[1], " & ", lpolarhalley[3], " & ", lpolarhalley[4], "\\\\ \n")
    print("QDWH~\\cite{NakatsukasaBaiGygi2010} & ", lpolarqr[1], " & ", lpolarqr[3], " & ", lpolarqr[4], "\\\\ \n")
    print("CDWH~\\cite{NakatsukasaBaiGygi2010} & ", lpolarchol[1], " & ", lpolarchol[3], " & ", lpolarchol[4], "\\\\ \n")
    print("Q-CDWH & ", lpolarhybrid[1], " & ", lpolarhybrid[3], " & ", lpolarhybrid[4], "\\\\ \n")
end
println("\\hline")