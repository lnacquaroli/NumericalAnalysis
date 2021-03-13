### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ e340b930-813d-11eb-049d-0fbafc60775d
using Compat

# ╔═╡ c39e460e-813d-11eb-36fe-c516f2c4860e
md"### Orthogonalization by Householder reflectors method.

Chapter 4, Sauer's Numerical Analysis.
"

# ╔═╡ e6e3a980-813d-11eb-2cdd-af5a7636288d
@compat import SparseArrays as spars

# ╔═╡ af578250-81df-11eb-2d6c-697343432a73
@compat import LinearAlgebra as linalg

# ╔═╡ e6d74d70-813d-11eb-1a84-09eda3ffd0d0
"""
Orthogonalization by Householder reflectors

	sol = householder_reflectors(A)

A: m×n matrix with linearly independent columns (m≤n) 

sol: Full QR solution
	Q: orthogonal matrix (m×n, Qᵀ = Q^-1, preserves the Euclidean norm of a vector)
	R: upper triangular (m×n)
"""
function householder_reflectors(A)
	m, n = size(A)
	m ≥ n || throw(DimensionMismatch("The number of rows in A must be greater than or equal to the number of columns."))
	R = copy(A)
	Q = Array(1.0*linalg.I(m))
	H = [] # eye = H = Array(1.0*linalg.I(m))
	for i in 1:min(n,m-1)
		x = R[i:m,i]
		w = [sign(x[1])*linalg.norm(x,2); zeros(m-i,1)]
		v = vec(w .- x)
		H = Array(1.0*linalg.I(m)) # H = eye
		H[i:m,i:m] .= Array(1.0*linalg.I(m-i+1)) .- 2.0*v*v' ./ (v'*v)
		Q *= H
		R = H*R
	end
	return (Q=Q, R=R)
end

# ╔═╡ 37e6c260-8295-11eb-158d-a7886133d571
A = [3 1; 4 3]

# ╔═╡ 37a1f130-8295-11eb-2c07-cdcd34a0a9fa
sol = householder_reflectors(A)

# ╔═╡ f8be1c10-82a7-11eb-3539-0bda4a0e25a9
B = [1 -4; 2 3; 2 2]

# ╔═╡ f8ac68d0-82a7-11eb-0dc8-697e98483c9a
householder_reflectors(B)

# ╔═╡ f891dbf0-82a7-11eb-1c73-139a13dbfb7a


# ╔═╡ f8772800-82a7-11eb-2dff-15762ca3a750


# ╔═╡ Cell order:
# ╟─c39e460e-813d-11eb-36fe-c516f2c4860e
# ╠═e340b930-813d-11eb-049d-0fbafc60775d
# ╠═e6e3a980-813d-11eb-2cdd-af5a7636288d
# ╠═af578250-81df-11eb-2d6c-697343432a73
# ╠═e6d74d70-813d-11eb-1a84-09eda3ffd0d0
# ╠═37e6c260-8295-11eb-158d-a7886133d571
# ╠═37a1f130-8295-11eb-2c07-cdcd34a0a9fa
# ╠═f8be1c10-82a7-11eb-3539-0bda4a0e25a9
# ╠═f8ac68d0-82a7-11eb-0dc8-697e98483c9a
# ╠═f891dbf0-82a7-11eb-1c73-139a13dbfb7a
# ╠═f8772800-82a7-11eb-2dff-15762ca3a750
