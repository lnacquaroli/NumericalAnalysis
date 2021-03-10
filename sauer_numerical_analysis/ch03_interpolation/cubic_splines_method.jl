### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ ceadac40-8105-11eb-15de-a78b56f517a5
using Compat

# ╔═╡ b4342d30-8105-11eb-1a09-0fca46754878
md"### Cubic splines method.

Chapter 3, Sauer's Numerical Analysis.
"

# ╔═╡ ce978c30-8105-11eb-01a2-895299ba10e2
@compat import Plots as plt

# ╔═╡ 2a0826a0-810c-11eb-10d0-f52ea82e05ba
@compat import SparseArrays as spars

# ╔═╡ 3c987270-8107-11eb-2100-db1bf0edc387
function _check_sorted(x)
	for i in 1:length(x)-1, j in i+1:length(x)
		if x[i] > x[j]
			error("The x-vector must be sorted in ascendant fashion (x[i]<x[i+1]).")
		end
	end
end

# ╔═╡ 0166a6a0-810b-11eb-21d8-93df096d2b79
function _set_endpoints!!(A, r, ep, δ, Δ, v)
	n = length(r)
	# Set end-points
	if ep == :natural
		A[1,1] = 1.0
		A[n,n] = 1.0
	elseif ep == :curvature
		A[1,1] = 2.0
		r[1] = v[1]
		A[n,n] = 2.0
		r[n] = v[2]
	elseif ep == :clamped
		A[1,1] = 2.0*δ[1]
		A[1,2] = δ[1]
		r[1] = 3.0*(Δ[1]/δ[1] - v[1])
		A[n,n-1] = δ[n-1]
		A[n,n] = 2.0*δ[n-1]
		r[n] = 3.0*(v[2] - Δ[n-1]/δ[n-1])
	elseif ep == :parabolic
		A[1,1] = 1.0
		A[1,2] = -1.0
		A[n,n-1] = 1.0
		A[n,n] = -1.0
	elseif ep == :notaknot
		A[1,1] = δ[2]
		A[1,2] = -(δ[1] + δ[2])
		A[1,3] = δ[1]
		A[n,n-2] = δ[n-1]
		A[n,n-1]= -(δ[n-2] + δ[n-1]) 
		A[n,n] = δ[n-2]
	end
end

# ╔═╡ ce8563c0-8105-11eb-0bdd-bdd7e7f307cd
"""
Calculates coefficients of cubic spline

	C = cubic_spline_coefficients(x, y; ep=:natural, v=[0.0, 0.0], k=10)

x:  x-coordinate data points
y:  y-coordinate data points corresponding to x
ep: End-points conditions (:natural, :curvature, :clamped, :parabolic, :notaknot)
	ep = :natural, :curvature, :clamped, needs length(x) ≥ 2
	ep = :parabolic, needs length(x) ≥ 3
	ep = :notaknot, needs length(x) ≥ 4
v:  Vector containing the values of the end points for the :curvature and :clamped ep
k:  Number of points per segment (mostly for plotting)

C:  Matrix of coefficients 
	b1, c1, d1;
	b2, c2, d2;...
"""
function cubic_spline_coefficients(x, y; ep=:natural, v=[0.0, 0.0], k=20)
	n = length(x)
	n == length(y) || throw(DimensionMismatch("x and y must have the same length."))
	
	_check_sorted(x)
	
	δ = zeros(n-1)
	Δ = similar(δ)
	for i in 1:n-1
		δ[i] = x[i+1] - x[i]
		Δ[i] = y[i+1] - y[i]
	end
	
	A = spars.spzeros(n,n) # We can do a Triangular matrix as well with LA
	r = zeros(n) # right-hand side
	for i in 2:n-1
		# A[i, i-1:i+1] = [δ[i-1], 2.0*(δ[i-1] + δ[i]), δ[i]]
		A[i,i-1] = δ[i-1]
		A[i,i] = 2.0*(δ[i-1] + δ[i])
		A[i,i+1] = δ[i]
		r[i] = 3.0*(Δ[i]/δ[i] - Δ[i-1]/δ[i-1])
	end
	
	_set_endpoints!!(A, r, ep, δ, Δ, v)
	
	C = zeros(n,3) # Coefficients matrix C = [b c d]
	C[:,2] = A\r # Solve for c coefficients
	# Solve for b and d
	for i in 1:n-1
		C[i,3] = (C[i+1,2] - C[i,2])/(3.0*δ[i])
		C[i,1] = Δ[i]/δ[i] - δ[i]*(2.0*C[i,2] + C[i+1,2])/3.0
	end
	C = C[1:n-1,:]
	
	# Build the splines
	x1 = []
	y1 = []
	for i in 1:n-1
		xs = LinRange(x[i],x[i+1],k+1)
		δ = xs .- x[i]
		# Evaluate using nested multiplication
		ys = C[i,3].*δ;
		ys = (ys .+ C[i,2]).*δ
		ys = (ys .+ C[i,1]).*δ .+ y[i]
		x1 = [x1; vec(xs[1:k])]
		y1 = [y1; vec(ys[1:k])]
	end
	x1 = [x1; x[end]]
	y1 = [y1; y[end]]
	
	return (C=C, x=float.(x1), y=float.(y1))
end

# ╔═╡ ce4c5260-8105-11eb-1f9a-8d777781082f
x=[0, 1, 2, 3, 4, 5]

# ╔═╡ cdac68e2-8105-11eb-3001-57ddd7591559
y=[3, 1, 4, 1, 2, 0]

# ╔═╡ cd615620-8105-11eb-0b9e-11dc588aaa26
natural = cubic_spline_coefficients(x, y);

# ╔═╡ 48e45c2e-810f-11eb-0437-790cb7aac64e
nak = cubic_spline_coefficients(x, y; ep=:notaknot);

# ╔═╡ a2957b10-810f-11eb-279d-798bf53f05a8
parabolic = cubic_spline_coefficients(x, y; ep=:parabolic);

# ╔═╡ a27a7900-810f-11eb-322b-075156ec16a7
clamped = cubic_spline_coefficients(x, y; ep=:clamped);

# ╔═╡ 1078ee40-8111-11eb-067b-af7ef0a2f63c
curvature = cubic_spline_coefficients(x, y; ep=:curvature);

# ╔═╡ fa88963e-810f-11eb-11be-618aab31e9c8
plt.plot(natural.x, natural.y, label="Natural")
plt.plot!(nak.x, nak.y, label="Not-a-knot")
plt.plot!(parabolic.x, parabolic.y, label="Parabolic")
plt.plot!(clamped.x, clamped.y, label="Clamped, v=[0,0]")
plt.plot!(curvature.x, curvature.y, label="Curvature, v=[0,0]")
plt.scatter!(x, y, label="Data points")

# ╔═╡ fa4069b0-810f-11eb-0f79-037b5c6501e1


# ╔═╡ fa262af0-810f-11eb-19d3-85c5efa69218


# ╔═╡ Cell order:
# ╟─b4342d30-8105-11eb-1a09-0fca46754878
# ╠═ceadac40-8105-11eb-15de-a78b56f517a5
# ╠═ce978c30-8105-11eb-01a2-895299ba10e2
# ╠═2a0826a0-810c-11eb-10d0-f52ea82e05ba
# ╟─3c987270-8107-11eb-2100-db1bf0edc387
# ╠═0166a6a0-810b-11eb-21d8-93df096d2b79
# ╠═ce8563c0-8105-11eb-0bdd-bdd7e7f307cd
# ╠═ce4c5260-8105-11eb-1f9a-8d777781082f
# ╠═cdac68e2-8105-11eb-3001-57ddd7591559
# ╠═cd615620-8105-11eb-0b9e-11dc588aaa26
# ╠═48e45c2e-810f-11eb-0437-790cb7aac64e
# ╠═a2957b10-810f-11eb-279d-798bf53f05a8
# ╠═a27a7900-810f-11eb-322b-075156ec16a7
# ╠═1078ee40-8111-11eb-067b-af7ef0a2f63c
# ╠═fa88963e-810f-11eb-11be-618aab31e9c8
# ╠═fa4069b0-810f-11eb-0f79-037b5c6501e1
# ╠═fa262af0-810f-11eb-19d3-85c5efa69218
