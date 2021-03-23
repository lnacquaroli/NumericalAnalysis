### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 90cd7b30-859c-11eb-1093-235164b20800
using Compat

# ╔═╡ 7ba6bf50-859c-11eb-071c-613e56710df5
md"### Generalized Minimum Residual Method (GMRES).

Chapter 4, Sauer's Numerical Analysis.

"

# ╔═╡ 90b5fb90-859c-11eb-30b6-756568ef10cf
@compat import SparseArrays as spars

# ╔═╡ 9099e810-859c-11eb-28b9-5b7ea02e7acc
@compat import LinearAlgebra as linalg

# ╔═╡ 378e2680-8684-11eb-3341-99ef476e5322
@compat import Plots as plt

# ╔═╡ 6b18f740-85c2-11eb-2eea-757f5276aa07
"""
Compute the Givens rotation matrix parameters for a and b.
ref: http://www.netlib.org/templates/matlab/rotmat.m
"""
function givens_rotation(a, b)
	if b == 0.0
		c, s = 1.0, 0.0
	elseif abs(b) > abs(a)
		temp = a/b
		s = 1.0 / sqrt(1.0 + temp^2)
		c = temp*s
	else
		temp = b/a
		c = 1.0 / sqrt(1.0 + temp^2)
		s = temp*c
	end
	return c, s
end

# ╔═╡ 06383702-865d-11eb-0c1b-357488bbb20c
# Modified Gram-Schmidt orthonormalization to a Krylov space (Arnoldi method)
function arnoldi!(H, Q, M, i)
	w = M \ (A*Q[:,i])
	for k in 1:i
		H[k,i] = transpose(w)*Q[:,k]
		w = w .- H[k,i]*Q[:,k]
	end
	H[i+1,i] = linalg.norm(w,2)
	Q[:,i+1] = vec(w ./ H[i+1,i])
end

# ╔═╡ 4124ca2e-865e-11eb-0ab5-b3db8c3bae43
"""
Givens rotation for the H matrix
Eliminate the last element in H ith row and update the rotation matrix
"""
function rotate_matrix!(H, β, cs, sn, i)
	for k in 1:i-1
		H[k,i], H[k+1,i] = cs[k]*H[k,i]+sn[k]*H[k+1,i], -sn[k]*H[k,i]+cs[k]*H[k+1,i]
		# temp = cs[k]*H[k,i] + sn[k]*H[k+1,i]
		# H[k+1,i] = -sn[k]*H[k,i] + cs[k]*H[k+1,i]
		# H[k,i] = temp
	end
	cs[i], sn[i] = givens_rotation(H[i,i], H[i+1,i]) # form i-th rotation matrix
	# Approximate residual norm
    β[i+1] = -sn[i]*β[i]
	β[i] = cs[i]*β[i]
	H[i,i] = cs[i]*H[i,i] + sn[i]*H[i+1,i]
	H[i+1,i] = 0.0
end

# ╔═╡ 2b2fe910-8683-11eb-2258-eb56e63b6a80
function update_approximation!(x, H, Q, β, k)
	# @show H[1:k,1:k]
	# @show β[1:k]
	c = H[1:k,1:k] \ β[1:k]
	x .+= Q[:,1:k]*c
end

# ╔═╡ 9083a0ee-859c-11eb-1ce0-05d9614572db
"""
Preconditioned Generalized Minimum Residual Method (GMRES) for solving non-symmetric Ax=b systems.

The process uses the Givens rotations to transform the matrix H and the residuals accordingly.

	sol = gmres_method(A, b, x₀; M=1.0*linalg.I(length(b)), steps=10, tol=1e-6)

A:     Coefficients matrix (m×n)
b:     rhs vector (m)
x₀:    Vector with initial guess (n)
M:     Conditioning matrix
steps: Number of iterations to run
tol:   Tolerance. Stopping criteria

sol: Solution
	x:    Best approximation
	Q:    Orthonormal matrix (m×k), k-dimensional Krylov space 
	H:    Upper Hassengber matrix (k+1, k) (sparse)
	r:    Vector with residuals
	iter: Number of iterations run
	err:  Error per iteration

ref: https://en.wikipedia.org/wiki/Generalized_minimal_residual_method#Regular_GMRES_(MATLAB_/_GNU_Octave)
"""
function gmres_method(A, b, x₀; M=1.0*linalg.I(length(b)), steps=10, tol=1e-6)
	bnorm2 = linalg.norm(b,2)
	bnorm2 == 0.0 ? bnorm2 = 1.0 : nothing
	
	x = copy(x₀)
	r = M \ (b .- A*x) # initial residual
	ε = linalg.norm(r,2) / bnorm2
	if ε ≤ tol; return (x=x, Q=[], H=[], r=r, iter=0, err=ε); end
	
	n, nc = size(A)
	n ≥ nc || throw(DimensionMismatch("The number of rows in A must be greater than or equal to the number of columns."))
	
	# Initialize storage
	m = copy(steps)
	Q = zeros(n, steps+1)
	H = spars.spzeros(m+1, m)
	cs = zeros(m)
	sn = zeros(m)
	e₁ = [1.0; zeros(n-1)]
	β = vec(linalg.norm(r,2)*e₁)
	
	Q[:,1] .= vec(r ./ linalg.norm(r,2))
	k = 0
	# ε = [linalg.norm(r,2)/bnorm2]
	ε = []
	while k < steps
		
		k += 1
		
		# Arnoldi method
		arnoldi!(H, Q, M, k)
		
		# Givens rotation for the H matrix and residual β
		rotate_matrix!(H, β, cs, sn, k)
		
		# Store error
		push!(ε, abs(β[k+1]/bnorm2))
		ε[k] ≤ tol && break
		
		# Update approximation
		update_approximation!(x, H, Q, β, k)
		
	end
	
	return (x=x, Q=Q, H=H, r=r, iter=k, err=float.(ε))
end

# ╔═╡ c560943e-8687-11eb-3eaf-ff1b44929021
"""
Restarted Preconditioned Generalized Minimum Residual Method (GMRES) for solving non-symmetric Ax=b systems.

The process uses the Givens rotations to transform the matrix H and the residuals accordingly.

	sol = restarted_gmres_method(A, b, x₀; M=1.0*linalg.I(length(b)), steps=20, tol=1e-6, rstart=5)

A:      Coefficients matrix (m×n)
b:      rhs vector (m)
x₀:     Vector with initial guess (n)
M:      Conditioning matrix
steps:  Number of iterations to run
tol:    Tolerance. Stopping criteria
rstart: Iteration at which the process restarts

sol: Solution
	x:    Best approximation
	Q:    Orthonormal matrix (m×k), k-dimensional Krylov space 
	H:    Upper Hassengber matrix (k+1, k) (sparse)
	r:    Vector with residuals
	iter: Number of iterations run
	err:  Error per iteration
	msg:  Convergence message

ref: http://www.netlib.org/templates/matlab/gmres.m
"""
function restarted_gmres_method(A, b, x₀; M=1.0*linalg.I(length(b)), steps=20, tol=1e-6, rstart=5)
	bnorm2 = linalg.norm(b,2)
	bnorm2 == 0.0 ? bnorm2 = 1.0 : nothing
	
	x = copy(x₀)
	r = M \ (b .- A*x) # initial residual
	ε = linalg.norm(r,2) / bnorm2
	if ε ≤ tol; return (x=x, Q=[], H=[], r=r, iter=0, err=ε); end
	
	n, nc = size(A)
	n ≥ nc || throw(DimensionMismatch("The number of rows in A must be greater than or equal to the number of columns."))
	
	# Initialize storage
	m = copy(rstart)
	Q = spars.zeros(n, steps+1)
	H = spars.spzeros(m+1, m)
	cs = zeros(m)
	sn = zeros(m)
	e₁ = [1.0; zeros(n-1)]
	
	k = 0
	ε = []
	while k < steps
		k += 1
		
		r = M \ (b .- A*x)
		β = vec(linalg.norm(r,2)*e₁)
		Q[:,1] .= vec(r ./ linalg.norm(r,2))
		
		for i in 1:m # restart iterations
			
			# Arnoldi method
			arnoldi!(H, Q, M, i)
			
			# Givens rotation for the H matrix and residual β
			rotate_matrix!(H, β, cs, sn, i)
			
			# Store error
			push!(ε, abs(β[i+1]/bnorm2))
			ε[i] ≤ tol && update_approximation!(x, H, Q, β, i) && break
			
		end
		
		ε[k] ≤ tol && break
		update_approximation!(x, H, Q, β, m)
		
		r = M \ (b .- A*x)
		β[k+1] = linalg.norm(r,2)
		push!(ε, abs(β[k+1]/bnorm2))
		
		ε[k] ≤ tol && break
		
	end
	
	msg = ε[end] > tol ? "Not converged." : "Converged."
	
	return (x=x, Q=Q, H=H, r=r, iter=(steps=k,total=length(ε)), err=float.(ε), msg=msg)
end

# ╔═╡ 906c2150-859c-11eb-3047-8b8e8714dcb8
n = 500

# ╔═╡ 7ade0a50-865b-11eb-2da0-ef67eddebe86
A = spars.sparse(linalg.diagm(0 => sqrt.(1:n), 10 => cos.(1:(n-10)), -10 => sin.(1:(n-10))));

# ╔═╡ d1defd50-865b-11eb-36ac-436f2240883a
x0 = ones(n);

# ╔═╡ 90542c80-859c-11eb-1e27-a1dedf7ce0f0
b = fill(1,n);

# ╔═╡ 903d7030-859c-11eb-16e1-e341c7df5b53
sol = gmres_method(A, b, x0; steps=50);

# ╔═╡ 2d71fbc0-8681-11eb-1737-efeba6a2246d
plt.plot(1:sol.iter, sol.err)

# ╔═╡ 7cd18e62-8686-11eb-05a1-49e298fa0e73
sol.x

# ╔═╡ 7cbb6e4e-8686-11eb-1f8c-8f24e61252a3
sol.err[end]

# ╔═╡ 7ca52730-8686-11eb-3809-4fde584dffe2
sol2 = restarted_gmres_method(A, b, x0; steps=50, rstart=5);

# ╔═╡ a4eff450-868a-11eb-0d33-73c4f75c425a
plt.plot(1:sol2.iter.total, sol2.err)

# ╔═╡ a4c69a60-868a-11eb-0ed6-5db90518dcbe
length(sol2.err)

# ╔═╡ a4b3aea0-868a-11eb-2253-15adb2202d10
sol2.iter

# ╔═╡ a4812cee-868a-11eb-3e0a-3ff598f805b0
sol2.err[end]

# ╔═╡ 9e5dfa60-868a-11eb-0f92-c1c7783413fa
sol2.Q

# ╔═╡ Cell order:
# ╠═7ba6bf50-859c-11eb-071c-613e56710df5
# ╠═90cd7b30-859c-11eb-1093-235164b20800
# ╠═90b5fb90-859c-11eb-30b6-756568ef10cf
# ╠═9099e810-859c-11eb-28b9-5b7ea02e7acc
# ╠═378e2680-8684-11eb-3341-99ef476e5322
# ╟─6b18f740-85c2-11eb-2eea-757f5276aa07
# ╟─06383702-865d-11eb-0c1b-357488bbb20c
# ╟─4124ca2e-865e-11eb-0ab5-b3db8c3bae43
# ╠═2b2fe910-8683-11eb-2258-eb56e63b6a80
# ╟─9083a0ee-859c-11eb-1ce0-05d9614572db
# ╠═c560943e-8687-11eb-3eaf-ff1b44929021
# ╠═906c2150-859c-11eb-3047-8b8e8714dcb8
# ╠═7ade0a50-865b-11eb-2da0-ef67eddebe86
# ╠═d1defd50-865b-11eb-36ac-436f2240883a
# ╠═90542c80-859c-11eb-1e27-a1dedf7ce0f0
# ╠═903d7030-859c-11eb-16e1-e341c7df5b53
# ╠═2d71fbc0-8681-11eb-1737-efeba6a2246d
# ╠═7cd18e62-8686-11eb-05a1-49e298fa0e73
# ╠═7cbb6e4e-8686-11eb-1f8c-8f24e61252a3
# ╠═7ca52730-8686-11eb-3809-4fde584dffe2
# ╠═a4eff450-868a-11eb-0d33-73c4f75c425a
# ╠═a4c69a60-868a-11eb-0ed6-5db90518dcbe
# ╠═a4b3aea0-868a-11eb-2253-15adb2202d10
# ╠═a4812cee-868a-11eb-3e0a-3ff598f805b0
# ╠═9e5dfa60-868a-11eb-0f92-c1c7783413fa
