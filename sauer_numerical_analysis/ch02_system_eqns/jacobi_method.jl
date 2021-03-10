### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 453c2a70-7969-11eb-1ebb-61df7c314131
using Compat

# ╔═╡ c09bd590-7968-11eb-35bf-315fed7a6339
md"### Iterative methods: Jacobi's method

Chapter 2, Sauer's Numerical Analysis.
"

# ╔═╡ 465e1532-7969-11eb-0e1a-2942b045cd71
@compat import LinearAlgebra as linalg

# ╔═╡ df6e4e80-7968-11eb-0893-97912b14aaf9
"""
Implements the Jacobi method for solving the system Ax=b.
A must be square and strictly diagonally dominant.

	sol = jacobi_method(A, b, x₀; k=10, δ=1.0e-6)

A:  coefficient matrix (n × n)
b:  right hand side vector (n)
x₀: initial guess
k:  number of iterations
δ:  tolerance

sol:  solution
	x -> solution
	Dinv -> inverse of the diagonal of A
	L -> lower triangular with zeros in diagonal
	U -> upper triangular with zeros in diagonal
	err -> Absolute error between iteration
	iter -> number of iterations run
"""
function jacobi_method(A, b, x₀; k=20, δ=1.0e-6)
	m, n = size(A)
	m == n || throw("Matrix must be square.")
	
	# Take the inverse of the diagonal elements directly since its a vector
	Dinv = linalg.Diagonal(1.0 ./ linalg.diag(A))
	#D = linalg.diag(A)
	
	# Build L and U (not the same as LU factorization) with zeros in diagonal
	L = copy!(similar(A), linalg.LowerTriangular(A)) # otherwise modifies A in next line
	L[CartesianIndex.(1:m, 1:n)] .= 0.0
	U = copy!(similar(A), linalg.UpperTriangular(A)) # otherwise modifies A in next line
	U[CartesianIndex.(1:m, 1:n)] .= 0.0
	LU = L + U
	# Another form with remainder
	# LU = A .- D 
	
	x1 = copy!(similar(x₀), x₀)
	x2 = similar(x1)
	ε = Inf
	i = 0
	while (ε > δ) || (i < k)
		i += 1
		x2 = Dinv*(b - LU*x1)
		ε = linalg.norm((x2 - x1)/x1, Inf)
		x1 = x2
	end
	return (x=x2, Dinv=Dinv, L=L, U=U, err=ε, iter=i)
	#return (x=x2, Dinv=Dinv, LU=LU, err=ε, iter=i)
end

# ╔═╡ df0b6e00-7968-11eb-0849-fdfeb6911053
A = [3 1; 1 2]

# ╔═╡ def240b0-7968-11eb-3833-d7dd2e387538
b = [5; 5]

# ╔═╡ dedf2de0-7968-11eb-1363-a91545b244af
sol = jacobi_method(A, b, [0, 0])

# ╔═╡ 8efd8710-7970-11eb-304e-b503aebfdd95
linalg.dot

# ╔═╡ 8ecc8c00-7970-11eb-09f8-07a60dedf726
"""
Implements the Jacobi method for solving the system Ax=b.
A must be square and strictly diagonally dominant.

	sol = jacobi_method_matrix_free(A, b, x₀; k=10, δ=1.0e-6)

A:  coefficient matrix (n × n)
b:  right hand side vector (n)
x₀: initial guess
k:  number of iterations
δ:  tolerance

sol:  solution
	x -> solution
	err -> Absolute error between iteration
	iter -> number of iterations run
"""
function jacobi_method_matrix_free(A, b, x₀; k=20, δ=1.0e-6)
	m, n = size(A)
	m == n || throw("Matrix must be square.")
	
	x1 = copy!(similar(x₀), x₀)
	x2 = similar(x1)
	ε = Inf
	i = 0
	while (ε > δ) || (i < k)
		i += 1
		for r in 1:m
			s = 0.0
			for c in 1:n
				if c != r
					s += A[r, c] * x1[c]
				end
			end
			x2[r] = 1.0 / A[r,r] * (b[r] - s)
		end
		ε = linalg.norm((x2 - x1)/x1, Inf)
		x1 = x2
	end
	return (x=x2, err=ε, iter=i)
end

# ╔═╡ 8eb162de-7970-11eb-1027-697b34576cd9
sol2 = jacobi_method_matrix_free(float.(A), float.(b), zeros(2))

# ╔═╡ 69ae3e50-797a-11eb-0b45-45674930586b
@time jacobi_method_matrix_free(rand(10,10), rand(10), zeros(10))

# ╔═╡ aa32f78e-797a-11eb-21d8-f70dd901ae45
@time jacobi_method(rand(10,10), rand(10), zeros(10))

# ╔═╡ Cell order:
# ╟─c09bd590-7968-11eb-35bf-315fed7a6339
# ╠═453c2a70-7969-11eb-1ebb-61df7c314131
# ╠═465e1532-7969-11eb-0e1a-2942b045cd71
# ╠═df6e4e80-7968-11eb-0893-97912b14aaf9
# ╠═df0b6e00-7968-11eb-0849-fdfeb6911053
# ╠═def240b0-7968-11eb-3833-d7dd2e387538
# ╠═dedf2de0-7968-11eb-1363-a91545b244af
# ╠═8efd8710-7970-11eb-304e-b503aebfdd95
# ╠═8ecc8c00-7970-11eb-09f8-07a60dedf726
# ╠═8eb162de-7970-11eb-1027-697b34576cd9
# ╠═69ae3e50-797a-11eb-0b45-45674930586b
# ╠═aa32f78e-797a-11eb-21d8-f70dd901ae45
