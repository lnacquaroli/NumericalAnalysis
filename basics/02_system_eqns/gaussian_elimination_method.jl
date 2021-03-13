### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ f546ae92-78a1-11eb-085b-0b5931cd4ca5
using Compat

# ╔═╡ c802da30-78a1-11eb-348c-153c7b96b1bc
md"### Gaussian elimination method

Chapter 2, Sauer's Numerical Analysis.
"

# ╔═╡ f5326340-78a1-11eb-0e7f-a52177723af2
@compat import LinearAlgebra as linalg

# ╔═╡ 9dc87110-78a8-11eb-35f9-db921bb156d4
@compat import SparseArrays as spars

# ╔═╡ f51df0e0-78a1-11eb-0859-b9cc9c3a4f8c
"""Eliminiation process"""
function elimination!(U, f, n)
	for j in 1:n-1
		abs(U[j,j]) > eps() || throw("Zero pivot encountered.")
		for i in j+1:n
			factor = U[i,j]/U[j,j]
			for k in j+1:n # j will be zero and we are not coming back to it
				U[i,k] = U[i,k] - factor*U[j,k]
			end
			f[i] = f[i] - factor*f[j]
		end
	end
end

# ╔═╡ f50a68de-78a1-11eb-371d-25a85ced79d8
"""Back-substitution process"""
function back_substitution!(x, f, U, n)
	for i in reverse(1:n)
		for j = i+1:n
			f[i] = f[i] - U[i,j]*x[j]
		end
	x[i] = f[i]/U[i,i]
	end
end

# ╔═╡ f4f58150-78a1-11eb-37dd-21d2b87cff32
"""
Perform naive Gaussian elimination on system Ax = b.
It has a stopper in case pivots are zero.

Input:
	A: Coefficient matrix (square)
	b: Vector with right-hand side (rhs) of equations

Output:
	x: back-substitution solutions
"""
function naive_gaussian_elimination(A, b)
	m, n = size(A)
	m == n || throw("Coefficient matrix is not square.")
	m == length(b) || throw("The size of the coefficient matrix and rhs vector are inconsistent.")
	
	U, f = copy(A), copy(b)
	elimination!(U, f, n)
	
	x = Vector{eltype(f[1])}(undef,n)
	back_substitution!(x, f, U, n)
	
	return x, linalg.UpperTriangular(U), f
end

# ╔═╡ f4e29590-78a1-11eb-3057-55eacb13e942
A = [1.0  2.0 -1.0; 2.0  1.0 -2.0; -3.0  1.0  1.0]

# ╔═╡ f4cd38d2-78a1-11eb-024b-9df94b892874
b = [3.0; 3.0; -6.0]

# ╔═╡ 2f4622a0-78a3-11eb-1cdc-77ba9e42c016
md"Perform naive Gaussian elimination:"

# ╔═╡ 304d74f0-78ad-11eb-2dae-5ff325f721f8
naive_gaussian_elimination(A, b)

# ╔═╡ 3020e6b0-78ad-11eb-269c-af7e63d03443


# ╔═╡ Cell order:
# ╟─c802da30-78a1-11eb-348c-153c7b96b1bc
# ╠═f546ae92-78a1-11eb-085b-0b5931cd4ca5
# ╠═f5326340-78a1-11eb-0e7f-a52177723af2
# ╠═9dc87110-78a8-11eb-35f9-db921bb156d4
# ╠═f51df0e0-78a1-11eb-0859-b9cc9c3a4f8c
# ╠═f50a68de-78a1-11eb-371d-25a85ced79d8
# ╠═f4f58150-78a1-11eb-37dd-21d2b87cff32
# ╠═f4e29590-78a1-11eb-3057-55eacb13e942
# ╠═f4cd38d2-78a1-11eb-024b-9df94b892874
# ╠═2f4622a0-78a3-11eb-1cdc-77ba9e42c016
# ╠═304d74f0-78ad-11eb-2dae-5ff325f721f8
# ╠═3020e6b0-78ad-11eb-269c-af7e63d03443
