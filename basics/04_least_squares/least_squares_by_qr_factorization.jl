### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ e340b930-813d-11eb-049d-0fbafc60775d
using Compat

# ╔═╡ c39e460e-813d-11eb-36fe-c516f2c4860e
md"### Least squares by QR factorization.

Chapter 4, Sauer's Numerical Analysis.
"

# ╔═╡ e6e3a980-813d-11eb-2cdd-af5a7636288d
@compat import SparseArrays as spars

# ╔═╡ af578250-81df-11eb-2d6c-697343432a73
@compat import LinearAlgebra as linalg

# ╔═╡ e6d74d70-813d-11eb-1a84-09eda3ffd0d0
"""
Classical Gram-Schmidt orthogonalization

	sol = classical_gram_schmidth(A)

A: m×n matrix with linearly independent (non-singular) columns (m≤n) 

sol: Reduced QR solution
	Q: orthogonal matrix (m×n, Qᵀ = Q^-1, preserves the Euclidean norm of a vector)
	R: upper triangular (m×n)
"""
function classical_gram_schmidth(A)
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
			R[i,j] = transpose(Q[:,i])*A[:,j]
			y = y .- R[i,j]*Q[:,i] 
		end
		R[j,j] = linalg.norm(y)
		Q[:,j] .= y ./ R[j,j]
	end
	return (Q=Q, R=R)
end

# ╔═╡ 3a658460-829d-11eb-1155-df78c68b2a90
"""Back-substitution process"""
function back_substitution!(x, f, U, n)
	for i in reverse(1:n)
		for j = i+1:n
			f[i] = f[i] - U[i,j]*x[j]
		end
	x[i] = f[i]/U[i,i]
	end
end

# ╔═╡ e0a4fc30-829c-11eb-20c1-8363464f6461
function least_squares_qr(A, b)
	m, n = size(A)
	qr = classical_gram_schmidth(A)
	R̂ = qr.R[1:n, 1:n]
	d̂ = (transpose(qr.Q)*vec(b))[1:n]
	x = similar(d̂)
	back_substitution!(x, d̂, R̂, n)
	return x
end

# ╔═╡ 37e6c260-8295-11eb-158d-a7886133d571
A = [1 -4; 2 3; 2 2]

# ╔═╡ a1648d52-829d-11eb-07c0-c7b10b4bb3df
b = [-3; 15; 9]

# ╔═╡ 37a1f130-8295-11eb-2c07-cdcd34a0a9fa
sol = classical_gram_schmidth(A)

# ╔═╡ 4707da00-85af-11eb-31ef-d9835e040fae
A \ b

# ╔═╡ 9ca30da0-829d-11eb-2350-d91b59de0a97
sol = least_squares_qr(A, b)

# ╔═╡ e5b45850-829d-11eb-0cb6-11f8ce9bab2a
x = (2 .+ (0:10) ./ 5)

# ╔═╡ e59f97d0-829d-11eb-0f05-2b9833154da0
y = @. 1+x+x^2+x^3+x^4+x^5+x^6+x^7

# ╔═╡ e5886650-829d-11eb-3008-ddfb172932a4
B = [x.^0 x x.^2 x.^3 x.^4 x.^5 x.^6 x.^7]

# ╔═╡ e5721f2e-829d-11eb-0899-99764e1b180b
qr = linalg.qr(B)

# ╔═╡ e55aedb0-829d-11eb-29f4-cde2812fdbcf
b = qr.Q'*y

# ╔═╡ e54283b0-829d-11eb-128c-4b3bce04a57c
c = qr.R[1:8,1:8]\b[1:8]

# ╔═╡ e528ba20-829d-11eb-1846-678317ee8207


# ╔═╡ Cell order:
# ╟─c39e460e-813d-11eb-36fe-c516f2c4860e
# ╠═e340b930-813d-11eb-049d-0fbafc60775d
# ╠═e6e3a980-813d-11eb-2cdd-af5a7636288d
# ╠═af578250-81df-11eb-2d6c-697343432a73
# ╠═e6d74d70-813d-11eb-1a84-09eda3ffd0d0
# ╠═3a658460-829d-11eb-1155-df78c68b2a90
# ╠═e0a4fc30-829c-11eb-20c1-8363464f6461
# ╠═37e6c260-8295-11eb-158d-a7886133d571
# ╠═a1648d52-829d-11eb-07c0-c7b10b4bb3df
# ╠═37a1f130-8295-11eb-2c07-cdcd34a0a9fa
# ╠═4707da00-85af-11eb-31ef-d9835e040fae
# ╠═9ca30da0-829d-11eb-2350-d91b59de0a97
# ╠═e5b45850-829d-11eb-0cb6-11f8ce9bab2a
# ╠═e59f97d0-829d-11eb-0f05-2b9833154da0
# ╠═e5886650-829d-11eb-3008-ddfb172932a4
# ╠═e5721f2e-829d-11eb-0899-99764e1b180b
# ╠═e55aedb0-829d-11eb-29f4-cde2812fdbcf
# ╠═e54283b0-829d-11eb-128c-4b3bce04a57c
# ╠═e528ba20-829d-11eb-1846-678317ee8207
