### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ e340b930-813d-11eb-049d-0fbafc60775d
using Compat

# ╔═╡ c39e460e-813d-11eb-36fe-c516f2c4860e
md"### Modified Gram-Schmidt orthogonalization method.

Chapter 4, Sauer's Numerical Analysis.
"

# ╔═╡ e6e3a980-813d-11eb-2cdd-af5a7636288d
@compat import SparseArrays as spars

# ╔═╡ af578250-81df-11eb-2d6c-697343432a73
@compat import LinearAlgebra as linalg

# ╔═╡ e6d74d70-813d-11eb-1a84-09eda3ffd0d0
"""
Modified Gram-Schmidt orthogonalization

	sol = modified_gram_schmidth(A)

A: m×n matrix with linearly independent columns (m≤n) 

sol: Full QR solution
	Q: orthogonal matrix (m×n, Qᵀ = Q^-1, preserves the Euclidean norm of a vector)
	R: upper triangular (m×n)
"""
function modified_gram_schmidth(A)
	m, n = size(A)
	m ≥ n || throw(DimensionMismatch("The number of rows in A must be greater than or equal to the number of columns."))
	if m != n
		I = 1.0linalg.I(m)
		A = [A Array(I[:, m-n])] # add m-n columns for full QR
	end
	m, n = size(A)
	R = spars.spzeros(m,n)
	Q = zeros(m,m)
	for j in 1:n
		y = A[:,j]
		for i in 1:j-1 # starts at j=2
			R[i,j] = transpose(Q[:,i])*y
			y = y .- R[i,j]*Q[:,i] 
		end
		R[j,j] = linalg.norm(y)
		Q[:,j] .= y ./ R[j,j]
	end
	return (Q=Q, R=R)
end

# ╔═╡ 37e6c260-8295-11eb-158d-a7886133d571
A = [1 -4; 2 3; 2 2]

# ╔═╡ 37a1f130-8295-11eb-2c07-cdcd34a0a9fa
sol = modified_gram_schmidth(A)

# ╔═╡ 9615dd5e-8297-11eb-3ffb-df8171d8697f
B = [1 -4 -1; 2 3 0; 2 2 0]

# ╔═╡ 95fd9a70-8297-11eb-1856-5b1bf28dc596
modified_gram_schmidth(B)

# ╔═╡ 95d13340-8297-11eb-2294-e5dafe9de472


# ╔═╡ 95b5e310-8297-11eb-0190-1f01a2d18585


# ╔═╡ Cell order:
# ╠═c39e460e-813d-11eb-36fe-c516f2c4860e
# ╠═e340b930-813d-11eb-049d-0fbafc60775d
# ╠═e6e3a980-813d-11eb-2cdd-af5a7636288d
# ╠═af578250-81df-11eb-2d6c-697343432a73
# ╠═e6d74d70-813d-11eb-1a84-09eda3ffd0d0
# ╠═37e6c260-8295-11eb-158d-a7886133d571
# ╠═37a1f130-8295-11eb-2c07-cdcd34a0a9fa
# ╠═9615dd5e-8297-11eb-3ffb-df8171d8697f
# ╠═95fd9a70-8297-11eb-1856-5b1bf28dc596
# ╠═95d13340-8297-11eb-2294-e5dafe9de472
# ╠═95b5e310-8297-11eb-0190-1f01a2d18585
