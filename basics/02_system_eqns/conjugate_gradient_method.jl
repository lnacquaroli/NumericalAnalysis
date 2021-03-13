### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ c0644cd0-7bc3-11eb-27ea-cd4e3d0c05f9
using Compat

# ╔═╡ 956bc7fe-7bc3-11eb-2d89-d3be54c356de
md"### Conjugate gradient method.

Chapter 2, Sauer's Numerical Analysis. Solving symmetric and positive-definite systems.
"

# ╔═╡ c04c09e0-7bc3-11eb-0c87-557a5156ec1b
@compat import LinearAlgebra as linalg

# ╔═╡ c03092a0-7bc3-11eb-091e-6db5b0d66d13
@compat import SparseArrays as spars

# ╔═╡ 4cfe40f0-7bcf-11eb-11d3-03c128a7508d
"""
Conjugate Gradient Method to solve the symmetric positive-definite system Ax = b.

	sol = conjugate_gradient(A, b, x₀; k=10, ε=1.0e-8)

A:  Symmetric and positive-definite matrix (size m × m).
b:  Right hand side vector (column, size m)
x₀: Initial guess (column, size m)
k:  Maximum number of iterations
ε:  Absolute error of the residual. Stopping criteria.

sol: Solution
	x:    x-solution vector
	res:  residuals of the solution
	dir:  search direction
	iter: number of iterations run
"""
function conjugate_gradient(A, b, x₀; k=10, ε=1.0e-8)
	m, _ = size(A)
	linalg.issymmetric(A) || throw("Input matrix must be symmetric.")
	all(linalg.eigvals(A) .> 0.0) || throw("Input matrix is not positive-definite.")
	
	d = r = vec(b) - A*vec(x₀)
	x = vec(copy(x₀))
	y, res, dir = [], [], []
	push!(y, x); push!(dir, d); push!(res, r)
	for _ in 1:k
		all(.!isapprox.(r, 0.0, atol=ε)) || return (x=y, res=res, dir=dir, iter=length(x)) 
		rr = transpose(r)*r
		α = rr/(transpose(d)*A*d)
		x = x .+ α.*d
		r = r .- α*A*d
		@show r
		β = transpose(r)*r ./ rr
		d = r .+ β.*d
		push!(y, x); push!(res, r); push!(dir, d)
	end
	return (x=y, res=res, dir=dir, iter=length(x)) 
end

# ╔═╡ c1b476ae-7bc7-11eb-0e43-7d48ffe6cc8b
B = [2.0 2.0; 2.0 5.0]

# ╔═╡ c19fdd3e-7bc7-11eb-110f-a38fd4a994a8
b = [6.0; 3.0]

# ╔═╡ c18bb900-7bc7-11eb-196b-f7f178a53d0a
sol = conjugate_gradient(B, b, [0.0; 0.0]; k=5)

# ╔═╡ be14de8e-7bc3-11eb-1a7e-736ff3c5a35a


# ╔═╡ Cell order:
# ╟─956bc7fe-7bc3-11eb-2d89-d3be54c356de
# ╠═c0644cd0-7bc3-11eb-27ea-cd4e3d0c05f9
# ╠═c04c09e0-7bc3-11eb-0c87-557a5156ec1b
# ╠═c03092a0-7bc3-11eb-091e-6db5b0d66d13
# ╠═4cfe40f0-7bcf-11eb-11d3-03c128a7508d
# ╠═c1b476ae-7bc7-11eb-0e43-7d48ffe6cc8b
# ╠═c19fdd3e-7bc7-11eb-110f-a38fd4a994a8
# ╠═c18bb900-7bc7-11eb-196b-f7f178a53d0a
# ╠═be14de8e-7bc3-11eb-1a7e-736ff3c5a35a
