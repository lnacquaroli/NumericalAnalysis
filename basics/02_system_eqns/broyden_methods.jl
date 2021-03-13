### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 5d8b0ef0-7d67-11eb-00f9-2933fd48671f
using Compat

# ╔═╡ becc6ca2-7d61-11eb-0d67-5fc765579ade
md"### Broyden’s Methods.

Chapter 2, Sauer's Numerical Analysis.
"

# ╔═╡ 610d7ef0-7d67-11eb-25c6-f1888bca50a1
@compat import LinearAlgebra as linalg

# ╔═╡ dfc5e440-7d61-11eb-0a68-85015784f2f3
"""
Finds multivariate roots using Broyden II (bad) method.
No derivatives, estimates the inverse of the estimate of the Jacobian, B.

	sol = broyden_method_II(F, x₀; B=1.0*linalg.I(length(x₀)), ε=1.0e-8, k=20)

F:  Function of several variables [F(u,v)]
x₀: Vector with initial guess
B:  Approximation of the jacobian (default I)
ε:  Absolute error in the change of successive x (stopping criteria)
k:  Maximum iterations (stopping criteria)

sol: Solution
	x:    Vector with solutions
	iter: Number of iterations run
    err:  Absolute error
"""
function broyden_method_II(F, x₀; B=1.0*linalg.I(length(x₀)), ε=1.0e-8, k=20)
	x0 = float.(copy(vec(x₀)))
	x1 = similar(x0)
	# m == n || throw("Coefficient matrix is not square.")
	# m == length(b) || throw("The size of the coefficient matrix and rhs vector are inconsistent.")
	for i = 1:k
		x1 = x0 .- B*F(x0...)
		δ = x1 .- x0
		Δ = F(x1...) - F(x0...)
		B = B + (δ - B*Δ)*transpose(δ)*B / (transpose(δ)*B*Δ)
		linalg.norm(x0.-x1, Inf) > ε || return (x=x0, iter=i, err=linalg.norm(x0.-x1, Inf))
		x0 = x1
	end
	return (x=x0, iter=k, err=linalg.norm(x0.-x1, Inf))
end

# ╔═╡ a6cd92d0-7e17-11eb-3dd4-1d1e4881c4be
"""
Finds multivariate roots using Broyden I (good) method.
No derivatives, estimates the inverse of the Jacobian with A.

	sol = broyden_method_II(F, x₀; A=1.0*linalg.I(length(x₀)), ε=1.0e-8, k=20)

F:  Function of several variables [F(u,v)]
x₀: Vector with initial guess
A:  Approximation of the jacobian (default I)
ε:  Absolute error in the change of successive x (stopping criteria)
k:  Maximum iterations (stopping criteria)

sol: Solution
	x:    Vector with solutions
	iter: Number of iterations run
    err:  Absolute error
"""
function broyden_method_I(F, x₀; A=1.0*linalg.I(length(x₀)), ε=1.0e-8, k=20)
	x0 = float.(copy(vec(x₀)))
	x1 = similar(x0)
	# m == n || throw("Coefficient matrix is not square.")
	# m == length(b) || throw("The size of the coefficient matrix and rhs vector are inconsistent.")
	for i = 1:k
		x1 = x0 .- linalg.pinv(A)*F(x0...)
		δ = x1 .- x0
		Δ = F(x1...) - F(x0...)
		A = A + (Δ - A*δ)*transpose(δ) / (transpose(δ)*δ)
		linalg.norm(x0.-x1, Inf) > ε || return (x=x0, iter=i, err=linalg.norm(x0.-x1, Inf))
		x0 = x1
	end
	return (x=x0, iter=k, err=linalg.norm(x0.-x1, Inf))
end

# ╔═╡ dfb171e0-7d61-11eb-1301-c3853feadf32
F(u,v) = [v - u^3; u^2 + v^2 - 1]

# ╔═╡ df9f7080-7d61-11eb-35be-33e2769fe7a5
J(u,v) = [-3*u^2 1; 2*u 2*v]

# ╔═╡ df8b2530-7d61-11eb-3b21-b3588a6de988
x0 = [1, 2]

# ╔═╡ df2f2280-7d61-11eb-157a-23cbd4c8182f
sol = broyden_method_II(F, x0; k=50)

# ╔═╡ 1677d3a0-7d66-11eb-11e1-c1da8a7ef851
sol2 = broyden_method_I(F, x0; k=50)

# ╔═╡ 162fce20-7d66-11eb-29b6-ffc095dbec6c


# ╔═╡ 1618eac0-7d66-11eb-0177-472553836f2c


# ╔═╡ 15fe5dde-7d66-11eb-1214-076a789d3017


# ╔═╡ 15e3f810-7d66-11eb-0371-b3e46f87e733


# ╔═╡ Cell order:
# ╟─becc6ca2-7d61-11eb-0d67-5fc765579ade
# ╠═5d8b0ef0-7d67-11eb-00f9-2933fd48671f
# ╠═610d7ef0-7d67-11eb-25c6-f1888bca50a1
# ╠═dfc5e440-7d61-11eb-0a68-85015784f2f3
# ╠═a6cd92d0-7e17-11eb-3dd4-1d1e4881c4be
# ╠═dfb171e0-7d61-11eb-1301-c3853feadf32
# ╠═df9f7080-7d61-11eb-35be-33e2769fe7a5
# ╠═df8b2530-7d61-11eb-3b21-b3588a6de988
# ╠═df2f2280-7d61-11eb-157a-23cbd4c8182f
# ╠═1677d3a0-7d66-11eb-11e1-c1da8a7ef851
# ╠═162fce20-7d66-11eb-29b6-ffc095dbec6c
# ╠═1618eac0-7d66-11eb-0177-472553836f2c
# ╠═15fe5dde-7d66-11eb-1214-076a789d3017
# ╠═15e3f810-7d66-11eb-0371-b3e46f87e733
