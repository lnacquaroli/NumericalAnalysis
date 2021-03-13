### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ c0644cd0-7bc3-11eb-27ea-cd4e3d0c05f9
using Compat

# ╔═╡ 956bc7fe-7bc3-11eb-2d89-d3be54c356de
md"### Preconditioned conjugate gradient method.

Chapter 2, Sauer's Numerical Analysis. Solving ill-conditioned systems.
"

# ╔═╡ c04c09e0-7bc3-11eb-0c87-557a5156ec1b
@compat import LinearAlgebra as linalg

# ╔═╡ c03092a0-7bc3-11eb-091e-6db5b0d66d13
@compat import SparseArrays as spars

# ╔═╡ a6bc51a0-7d4f-11eb-28f6-3b70ff0504df
function _precoditioned_matrix(A, m, pc, ω)
	D = linalg.Diagonal(A)
	if pc == :jacobi
		M = D
	elseif pc == :ssor || pc == :gaussseidel
		I = 1.0*linalg.I(m)
		LU = A .- D
		L = linalg.LowerTriangular(LU)
		U = linalg.UpperTriangular(LU)
		M = (I + ω.*L*linalg.pinv(D))*(D + ω.*U)
	end
	return M
end

# ╔═╡ 2b836000-7d5e-11eb-310d-050b6794f54f
function _preconditioned_matrix_2(A, m, ω)
	I = 1.0*linalg.I(m)
	d = linalg.diag(A)
	D = linalg.diagm(d)
	LU = A .- D
	L = linalg.LowerTriangular(LU)
	U = linalg.UpperTriangular(LU)
	Dinv = linalg.diagm(1.0 ./ d)
	M1 = I + ω.*L*Dinv
	M2 = D + ω.*U
	return M1, M2, Dinv
end

# ╔═╡ 1bd56d40-7d5b-11eb-0c2c-7532b90f2264
function _solution_Mz(Dinv, M1, M2, m, pc, r)
	if pc == :jacobi
		z = Dinv*r
	elseif pc == :ssor || pc == :gaussseidel
		c = similar(r)
		_back_sustitution!(c, copy(r), M1, m) 
		z = similar(r)
		_back_sustitution!(z, c, M2, m)
	end
	return z
end

# ╔═╡ 4cfe40f0-7bcf-11eb-11d3-03c128a7508d
"""
Preconditioned conjugate gradient Method for solving ill-conditioned systems Ax = b.

	sol = preconditioned_conjugate_gradient(A, b, x₀; k=10, ε=1.0e-8, pc=:jacobi, ω=1)

A:  Symmetric and positive-definite matrix (size m × m).
b:  Right hand side vector (column, size m)
x₀: Initial guess (column, size m)
k:  Maximum number of iterations
ε:  Absolute error of the residual. Stopping criteria.
pc: Preconditioner, :jacobi, :ssor, :gaussseidel
ω:  Symmetric SOR weight, 0 ≤ ω ≤ 2 (ω=1 => Gauss-Saidel)

sol: Solution
	x:    x-solution vector
	res:  residuals of the solution
	dir:  search direction
	z:    M^-1*res
	err:  norm-inf error of the residuals
 	iter: number of iterations run
"""
function preconditioned_conjugate_gradient(A, b, x₀; k=10, ε=1.0e-8, pc=:jacobi, ω=1)
	m, _ = size(A)
	linalg.issymmetric(A) || throw("Input matrix must be symmetric.")
	# all(linalg.eigvals(A) .> 0.0) || throw("Input matrix is not positive-definite.")
	
	M = _precoditioned_matrix(A, m, pc, ω)
	
	Minv = linalg.pinv(M)
	r = vec(b) - A*vec(x₀)
	d = z = Minv*r
	x = vec(copy(x₀))
	y, res, dir, zz = [], [], [], []
	push!(y, x); push!(dir, d); push!(res, r); push!(zz, z)
	i = 0
	δ = 0.0
	for _ in 1:k
		i += 1
		all(.!isapprox.(r, 0.0, atol=ε)) || return (x=y, res=res, dir=dir, zz=z, iter=i, err=linalg.norm(r,Inf)) 
		rz = transpose(r)*z
		α = rz/(transpose(d)*A*d)
		x = x .+ α.*d
		r = r .- α*A*d
		z = Minv*r
		β = transpose(r)*z ./ rz
		d = z .+ β.*d
		push!(y, x); push!(res, r); push!(dir, d); push!(zz, z)
	end
	return (x=y, res=res, dir=dir, zz=z, iter=i, err=linalg.norm(r,Inf))
end

# ╔═╡ e5c0ab70-7d5a-11eb-3177-53764d0a3bca
"""Back-sustitution process"""
function _back_sustitution!(x, f, U, n)
	for i in reverse(1:n)
		for j = i+1:n
			f[i] = f[i] - U[i,j]*x[j]
		end
	x[i] = f[i]/U[i,i]
	end
end

# ╔═╡ ed707172-7d5a-11eb-3c9d-3fcef0ba7634
"""
Preconditioned conjugate gradient Method for solving ill-conditioned systems Ax = b.

This version uses two back-sustitution instead of M^-1.

	sol = preconditioned_conjugate_gradient_2(A, b, x₀; k=10, ε=1.0e-8, pc=:jacobi, ω=1)

A:  Symmetric and positive-definite matrix (size m × m).
b:  Right hand side vector (column, size m)
x₀: Initial guess (column, size m)
k:  Maximum number of iterations
ε:  Absolute error of the residual. Stopping criteria.
pc: Preconditioner, :jacobi, :ssor, :gaussseidel
ω:  Symmetric SOR weight, 0 ≤ ω ≤ 2 (ω=1 => Gauss-Saidel)

sol: Solution
	x:    x-solution vector
	res:  residuals of the solution
	dir:  search direction
	z:    M^-1*res
 	iter: number of iterations run
	err:  norm-inf error of res
"""
function preconditioned_conjugate_gradient_2(A, b, x₀; k=10, ε=1.0e-8, pc=:jacobi, ω=1)
	m, _ = size(A)
	linalg.issymmetric(A) || throw("Input matrix must be symmetric.")
	# all(linalg.eigvals(A) .> 0.0) || throw("Input matrix is not positive-definite.")
	
	# Preconditioning matrix
	M1, M2, Dinv = _preconditioned_matrix_2(A, m, ω)
	
	r = vec(b) - A*vec(x₀)
	d = z = _solution_Mz(Dinv, M1, M2, m, pc, r)
	x = vec(copy(x₀))
	y, res, dir, zz = [], [], [], []
	push!(y, x); push!(dir, d); push!(res, r); push!(zz, z)
	i = 0
	for _ in 1:k
		i += 1
		all(.!isapprox.(r, 0.0, atol=ε)) || return (x=y, res=res, dir=dir, zz=z, iter=i, err=linalg.norm(r,Inf)) 
		rz = transpose(r)*z
		α = rz/(transpose(d)*A*d)
		x = x .+ α.*d
		r = r .- α*A*d
		z = _solution_Mz(Dinv, M1, M2, m, pc, r)
		β = transpose(r)*z ./ rz
		d = z .+ β.*d
		push!(y, x); push!(res, r); push!(dir, d); push!(zz, z)
	end
	return (x=y, res=res, dir=dir, zz=z, iter=i, err=linalg.norm(r,Inf))
end

# ╔═╡ fad972b0-7d5c-11eb-03e5-bd31d2daab83
r = vec(b) - A*zeros(n)

# ╔═╡ 2e1c76e0-7d5d-11eb-0d6e-9d5666e73362
_precoditioned_matrix_2(A, n, :jacobi, 1, r)

# ╔═╡ 998d9610-7d54-11eb-036e-5913b4646176
n = 500

# ╔═╡ c1b476ae-7bc7-11eb-0e43-7d48ffe6cc8b
A = linalg.diagm(0 => sqrt.(1:n), 10 => cos.(1:(n-10)), -10 => cos.(1:(n-10)));

# ╔═╡ c19fdd3e-7bc7-11eb-110f-a38fd4a994a8
b = A*ones(n);

# ╔═╡ b46c2d9e-7d56-11eb-371a-53c55927f26a
md"Jacobi preconditioner"

# ╔═╡ c18bb900-7bc7-11eb-196b-f7f178a53d0a
sol = preconditioned_conjugate_gradient(A, b, zeros(n); k=40, ε=eps());

# ╔═╡ ddbaa8c0-7d52-11eb-0c88-9382ed79ccfa
sol.iter, sol.err

# ╔═╡ dda991c0-7d52-11eb-3d81-95272f3a605f
sol.x[end]

# ╔═╡ bf654380-7d5c-11eb-281a-ad5609ae6340
sol_b = preconditioned_conjugate_gradient_2(A, b, zeros(n); k=40, ε=eps());

# ╔═╡ bf497e20-7d5c-11eb-3805-4b952f17537c
sol_b.iter, sol_b.err

# ╔═╡ d770cfd0-7d5c-11eb-11ca-e7be30078166
sol_b.x[end]

# ╔═╡ 3609a2c0-7d57-11eb-04df-5ba73e71542f
md"Gauss-Seidel preconditioner"

# ╔═╡ dd9657e2-7d52-11eb-36da-c356277f14c7
sol2 = preconditioned_conjugate_gradient(A, b, zeros(n); k=40, pc=:gaussseidel, ε=eps());

# ╔═╡ dd80d410-7d52-11eb-3780-dfd73c018f6e
sol2.iter, sol2.err

# ╔═╡ e905bcf0-7d5d-11eb-1c30-e7579ef32574
sol2_b = preconditioned_conjugate_gradient_2(A, b, zeros(n); k=40, pc=:gaussseidel, ε=eps());

# ╔═╡ f0357d80-7d5d-11eb-2929-6f06129c3c79
sol2_b.iter, sol2_b.err

# ╔═╡ 749683a0-7d57-11eb-35bf-07a26c84f1ab
md"SSOR preconditioner"

# ╔═╡ dd6d9a30-7d52-11eb-0a9f-af29c4a0ae39
sol3 = preconditioned_conjugate_gradient(A, b, zeros(n); k=40, ω=0.8, pc=:ssor, ε=eps());

# ╔═╡ dd5e7f00-7d52-11eb-3394-1325c6743f0d
sol3.iter, sol3.err

# ╔═╡ dd380b40-7d52-11eb-0b7f-a76ea09bf9e2
sol3_b = preconditioned_conjugate_gradient_2(A, b, zeros(n); k=40, ω=0.8, pc=:ssor, ε=eps());

# ╔═╡ dd1e68c0-7d52-11eb-1cda-c74f6aba66fc
sol3_b.iter, sol3_b.err

# ╔═╡ dcd21d80-7d52-11eb-3113-0bc6fa552af5


# ╔═╡ dcbbd660-7d52-11eb-26e2-2ba30bab1185


# ╔═╡ dca36c62-7d52-11eb-08fc-71ed595fee8b


# ╔═╡ dc79eb60-7d52-11eb-12a5-d91fa4e6e0c8


# ╔═╡ dc539eb2-7d52-11eb-2a74-694fa41a7d07


# ╔═╡ dc44837e-7d52-11eb-232e-bdd4125348f5


# ╔═╡ dc200b90-7d52-11eb-2d4e-356f8a8e597c


# ╔═╡ Cell order:
# ╟─956bc7fe-7bc3-11eb-2d89-d3be54c356de
# ╠═c0644cd0-7bc3-11eb-27ea-cd4e3d0c05f9
# ╠═c04c09e0-7bc3-11eb-0c87-557a5156ec1b
# ╠═c03092a0-7bc3-11eb-091e-6db5b0d66d13
# ╟─a6bc51a0-7d4f-11eb-28f6-3b70ff0504df
# ╠═2b836000-7d5e-11eb-310d-050b6794f54f
# ╠═1bd56d40-7d5b-11eb-0c2c-7532b90f2264
# ╠═4cfe40f0-7bcf-11eb-11d3-03c128a7508d
# ╠═e5c0ab70-7d5a-11eb-3177-53764d0a3bca
# ╠═ed707172-7d5a-11eb-3c9d-3fcef0ba7634
# ╠═fad972b0-7d5c-11eb-03e5-bd31d2daab83
# ╠═2e1c76e0-7d5d-11eb-0d6e-9d5666e73362
# ╠═998d9610-7d54-11eb-036e-5913b4646176
# ╠═c1b476ae-7bc7-11eb-0e43-7d48ffe6cc8b
# ╠═c19fdd3e-7bc7-11eb-110f-a38fd4a994a8
# ╠═b46c2d9e-7d56-11eb-371a-53c55927f26a
# ╠═c18bb900-7bc7-11eb-196b-f7f178a53d0a
# ╠═ddbaa8c0-7d52-11eb-0c88-9382ed79ccfa
# ╠═dda991c0-7d52-11eb-3d81-95272f3a605f
# ╠═bf654380-7d5c-11eb-281a-ad5609ae6340
# ╠═bf497e20-7d5c-11eb-3805-4b952f17537c
# ╠═d770cfd0-7d5c-11eb-11ca-e7be30078166
# ╠═3609a2c0-7d57-11eb-04df-5ba73e71542f
# ╠═dd9657e2-7d52-11eb-36da-c356277f14c7
# ╠═dd80d410-7d52-11eb-3780-dfd73c018f6e
# ╠═e905bcf0-7d5d-11eb-1c30-e7579ef32574
# ╠═f0357d80-7d5d-11eb-2929-6f06129c3c79
# ╠═749683a0-7d57-11eb-35bf-07a26c84f1ab
# ╠═dd6d9a30-7d52-11eb-0a9f-af29c4a0ae39
# ╠═dd5e7f00-7d52-11eb-3394-1325c6743f0d
# ╠═dd380b40-7d52-11eb-0b7f-a76ea09bf9e2
# ╠═dd1e68c0-7d52-11eb-1cda-c74f6aba66fc
# ╠═dcd21d80-7d52-11eb-3113-0bc6fa552af5
# ╠═dcbbd660-7d52-11eb-26e2-2ba30bab1185
# ╠═dca36c62-7d52-11eb-08fc-71ed595fee8b
# ╠═dc79eb60-7d52-11eb-12a5-d91fa4e6e0c8
# ╠═dc539eb2-7d52-11eb-2a74-694fa41a7d07
# ╠═dc44837e-7d52-11eb-232e-bdd4125348f5
# ╠═dc200b90-7d52-11eb-2d4e-356f8a8e597c
