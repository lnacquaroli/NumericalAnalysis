### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ c0644cd0-7bc3-11eb-27ea-cd4e3d0c05f9
using Compat

# ╔═╡ 956bc7fe-7bc3-11eb-2d89-d3be54c356de
md"### Cholesky factorization method.

Chapter 2, Sauer's Numerical Analysis. Solving symmetric and positive-definite systems.
"

# ╔═╡ c04c09e0-7bc3-11eb-0c87-557a5156ec1b
@compat import LinearAlgebra as linalg

# ╔═╡ c03092a0-7bc3-11eb-091e-6db5b0d66d13
@compat import SparseArrays as spars

# ╔═╡ c0167af0-7bc3-11eb-0712-fb22c6a476b0
"""
Standard algorithm for the Cholesky factorization to build the upper triangular matrix R, such that, given the input matrix A, A = RᵀR.

	R = cholesky_factorization(A)

A: Symmetric and positive-definite matrix.
R: Upper triangular matrix for factorization to solve Ax = b.

	1. Obtain R from Cholesky
	2. Solve auxiliary system, Rᵀc = b, to obtain c.
	3. Solve final system, Rx = c, to obtain x.
"""
function cholesky_factorization(U)
	A = copy(U)
	m, _ = size(A)
	linalg.issymmetric(A) || throw("Input matrix must be symmetric.")
	all(linalg.eigvals(A) .> 0.0) || throw("Input matrix is not positive-definite.")
	
	R = similar(A)
	for k in 1:m
		A[k,k] ≥ zero(A[k,k]) || throw("Diagonal element ($k, $k) is negative.")
		R[k,k] = sqrt(A[k,k])
		u = vec(A[k, k+1:m] ./ R[k,k])
		R[k,k+1:m] = u
		A[k+1:m,k+1:m] = A[k+1:m,k+1:m] .- u*transpose(u)
	end
	
	return linalg.UpperTriangular(R)
end

# ╔═╡ bfe558d0-7bc3-11eb-33b5-23c526e947e8
A = [4 -2 2; -2 2 -4; 2 -4 11]

# ╔═╡ bfd0e670-7bc3-11eb-3ff5-714bbd6908e2
cholesky_factorization(float.(A))

# ╔═╡ be14de8e-7bc3-11eb-1a7e-736ff3c5a35a


# ╔═╡ Cell order:
# ╟─956bc7fe-7bc3-11eb-2d89-d3be54c356de
# ╠═c0644cd0-7bc3-11eb-27ea-cd4e3d0c05f9
# ╠═c04c09e0-7bc3-11eb-0c87-557a5156ec1b
# ╠═c03092a0-7bc3-11eb-091e-6db5b0d66d13
# ╠═c0167af0-7bc3-11eb-0712-fb22c6a476b0
# ╠═bfe558d0-7bc3-11eb-33b5-23c526e947e8
# ╠═bfd0e670-7bc3-11eb-3ff5-714bbd6908e2
# ╠═be14de8e-7bc3-11eb-1a7e-736ff3c5a35a
