# SkewSchur.jl

Algorithms and experiments for skew-symmetric and structure-preserving Schur factorizations in Julia.

The repository is organized as a set of Julia source files and scripts rather than a packaged Julia project. The main routines focus on skew-symmetric matrices, polar decompositions, and related Schur-based factorizations.

## What is here

- `src/hamiltoschur.jl` implements `hamiltoschur` and `hamiltoschur!` for real skew-symmetric matrices.
- `src/polar.jl` contains the polar-decomposition routines used by the main algorithms: `polarqr!`, `polarchol!`, `polarhybrid!`, `polarhalley!`, `polarnewtonschulz!`, and `polarsvd!`.
- `src/other_schur.jl` provides alternative skew-symmetric Schur-related routines such as `householder_schur`, `hermitian_eigen`, and `randdiag`.
- `src/symplectoschur.jl` implements `symplectoschur` and `symplectoschur!` for special orthogonal matrices.
- `tests/` contains accuracy and timing scripts used to reproduce the figures and comparisons in the project.
- `applications/` contains example scripts and exploratory experiments.

## Basic use

Load the source files directly with `include` and call the routines on your matrix:

```julia
using LinearAlgebra, SkewLinearAlgebra

include("src/polar.jl")
include("src/hamiltoschur.jl")

A = randn(8, 8)
A = (A - A') / 2

Q, Σ = hamiltoschur(copy(A); polar = :hybrid)
```

For a real matrix Schur decomposition with the symplectic-style formulation:

```julia
include("src/symplectoschur.jl")

X = Matrix(qr(randn(8, 8).Q))
if det(X) < 0
    X[:, end] .*= -1
end
Q, C, S = symplectoschur(copy(X); skew = true, polar = :hybrid)
```

## Dependencies

The scripts use Julia standard libraries plus packages such as `SkewLinearAlgebra`, `Plots`, `BenchmarkTools`, and `LaTeXStrings` depending on the file you run. Install the needed packages in your Julia environment before executing the scripts.

## Reproducing results

- `tests/test_accuracy_hamiltoschur.jl` compares accuracy against other Schur-based methods.
- `tests/test_time_hamiltoschur.jl` measures timing for the main skew-symmetric Schur routines.
- `tests/test_accuracy_polar.jl` compares the polar-decomposition variants.
- `tests/test_accuracy_symplectoschur.jl` and `tests/test_time_symplectoschur.jl` benchmark the Schur routine for special orthogonal matrices.

The scripts in `applications/` show how the routines are used on concrete examples, including graph and matrix-structure experiments.

## Repository layout

```text
src/           Core algorithm implementations
tests/         Accuracy and timing scripts
applications/  Example workflows and exploratory scripts
figures/       Output location for generated plots
divers - bin/  Scratch scripts
```

## Notes

- The code is currently file-based; there is no `Project.toml` in the repository root.
- Most routines operate on copies of the input matrix and return the structured factors explicitly.
- Several scripts are written for experimentation and figure generation rather than as a stable public API.