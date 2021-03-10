### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 453c2a70-7969-11eb-1ebb-61df7c314131
using Compat

# ╔═╡ c09bd590-7968-11eb-35bf-315fed7a6339
md"### Iterative methods: Successive over-relaxation (SOR) method

Chapter 2, Sauer's Numerical Analysis.
"

# ╔═╡ 465e1532-7969-11eb-0e1a-2942b045cd71
@compat import LinearAlgebra as linalg

# ╔═╡ 8ecc8c00-7970-11eb-09f8-07a60dedf726
"""
Implements the Successive over-relaxation (SOR) method for solving the system Ax=b.
A must be square and strictly diagonally dominant.

	sol = sor_method(A, b, x₀; ω=1.0, k=10, δ=1.0e-6)

A:  coefficient matrix (n × n)
b:  right hand side vector (n)
x₀: initial guess
ω:  relaxation parameter (ω>1 -> SOR, ω<1 -> SUR)
k:  number of iterations
δ:  tolerance

sol:  solution
	x -> solution
	err -> Absolute error between iteration
	iter -> number of iterations run
"""
function sor_method(A, b, x₀; ω=1.0, k=10, δ=1.0e-6)
	m, n = size(A)
	m == n || throw("Matrix must be square.")
	
	x1 = copy!(similar(x₀), x₀)
	x2 = similar(x1)
	ε = Inf
	i = 0
	while (ε > δ) || (i < k)
		i += 1
		for r in 1:m
			s1, s2 = 0.0, 0.0
			for c in 1:r-1 # lower triangle
				s1 += A[r, c] * x2[c]
			end
			for c in r+1:n # upper triangle
				s2 += A[r, c] * x1[c] 
			end
			x2[r] = x1[r] * (1.0 - ω) +  ω / A[r,r] * (b[r] - s1 - s2)
		end
		ε = linalg.norm((x2 - x1)/x1, Inf)
		x1 = x2
	end
	return (x=x2, err=ε, iter=i)
end

# ╔═╡ 4d6ec5e0-7a36-11eb-3abb-359018668f71
A = [3 1 -1; 2 4 1; -1 2 5];

# ╔═╡ 5eef0f00-7a36-11eb-3946-85a720d3df05
b = [4; 1; 1];

# ╔═╡ 67b43ca0-7a36-11eb-12cc-79c4bd9d3083
sol = sor_method(float.(A), float.(b), zeros(3); ω=1.25)

# ╔═╡ da0e8760-7a36-11eb-19a8-89dd225a0734


# ╔═╡ Cell order:
# ╟─c09bd590-7968-11eb-35bf-315fed7a6339
# ╠═453c2a70-7969-11eb-1ebb-61df7c314131
# ╠═465e1532-7969-11eb-0e1a-2942b045cd71
# ╠═8ecc8c00-7970-11eb-09f8-07a60dedf726
# ╠═4d6ec5e0-7a36-11eb-3abb-359018668f71
# ╠═5eef0f00-7a36-11eb-3946-85a720d3df05
# ╠═67b43ca0-7a36-11eb-12cc-79c4bd9d3083
# ╠═da0e8760-7a36-11eb-19a8-89dd225a0734
