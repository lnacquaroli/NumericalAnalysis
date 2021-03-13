### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ a1da079e-807d-11eb-0bcc-9975938828c4
using Compat

# ╔═╡ 00615c20-807d-11eb-2e8c-e9b6bdd0fa71
md"### Newton divided difference interpolation method.

Chapter 3, Sauer's Numerical Analysis.
"

# ╔═╡ a1b73d60-807d-11eb-0cd2-d9ab26db9d20
@compat import SparseArrays as spars

# ╔═╡ 7ba588a0-807f-11eb-1bd6-f9e922d15b1d
@compat import Plots as plt

# ╔═╡ 0429c3e0-807f-11eb-2af0-dd6ee51f5445
"""
Nested multiplication
Evaluates polynomial from nested form using Horner’s Method

	y = horner_method(d, c, x; b=zeros(d))

d: Degree of polynomial
c: Array of d+1 coefficients (constant term first)
x: x-coordinate at which to evaluate
b: Array of d base points

y: Value of polynomial at x

"""
function horner_method(d, c, x; b=zeros(d))
	y = c[d+1]
	for i in d:-1:1
		y = @. y*(x - b[i]) + c[i]
	end
	return y
end

# ╔═╡ a1a710c0-807d-11eb-199c-c724488ea174
"""
Newton Divided Difference Interpolation Method. 
Computes coefficients of interpolating polynomial

	c = newton_dd(x,y)

x: vector with x-coordinate points (size = n)
y: vector containing y-coordinate points (size = n)

c: coefficients of interpolating polynomial in nested form (use with horner_method)
"""
function newton_dd(x, y)
	n = length(x)
	n == length(y) || throw(DimensionMismatch("x and y must have the same length."))
	
	length(unique(x)) !== length(x) && @warn("x-values are not unique, expect Inf in coefficients.")
	
	# Fill in y column of Newton triangle
	f = spars.spzeros(n,n)
	f[:,1] .= vec(y)
	
	# Fill in column from top to bottom
	for i in 2:n, j in 1:n+1-i
		f[j,i] = (f[j+1,i-1] - f[j,i-1])/(x[j+i-1] - x[j])
	end
	
	# Read along top of triangle for output coefficients
	# c = zeros(n)
	# for i in 1:n
	# 	c[i] = f[1,i]
	# end
	c = Array(f[1,:])
	
	return c, f
end

# ╔═╡ cc95df50-8082-11eb-38ac-f511ef23765f
"""
Building a sin function calculator
Approximates sin curve with degree 3 polynomial

	y = sine_function_polynomial_1(x)

x: values to evaluate the sin at

y: approximation for sin(x)
"""
function sine_function_polynomial_1(x)
	# Calculate the interpolating polynomial and store coefficients
	b = π*(0:3)/6.0 # b holds base points
	yb = sin.(b)
	c, _ = newton_dd(b, yb)
	
	# For each input x, move x to the fundamental domain and evaluate the
	# interpolating polynomial
	s = 1 # Correct the sign of sin
	x1 = mod(x, 2*π)
	if x1 > pi
		x1 = 2*π - x1
		s = -1
	end
	if x1 > π/2
		x1 = π - x1
	end
	
	# Evaluate the polynomial
	y = s*horner_method(3, c, x1; b=b)
	
	return y
end

# ╔═╡ a190c9a2-807d-11eb-3e22-95d363159acf
x0=[0, 2, 3]

# ╔═╡ a16c0390-807d-11eb-3350-836f45affd94
y0=[1, 2, 4]

# ╔═╡ a095c7ce-807d-11eb-1700-658e8d5d5e17
c, f = newton_dd(x0, y0)

# ╔═╡ a0828df0-807d-11eb-2d94-d7deea19e939
x = 0:.01:4

# ╔═╡ a06fc940-807d-11eb-0b2b-45ef5e3f3daf
y = horner_method(2, c, x; b=x0)

# ╔═╡ a04c14a0-807d-11eb-3a5d-e580178cd882
plt.scatter(x0, y0, label="Points")
plt.plot!(x, y, label="Polynomial")

# ╔═╡ c141db60-8080-11eb-27f7-bb7d4ac97a0b
sine_function_polynomial_1.([π/6, π/4, π/2, π])

# ╔═╡ 53009e80-8084-11eb-180b-616b7c6dfc3d
x1 = [1, 2, 3, 4, 14, 1000]

# ╔═╡ 32fd94d0-8084-11eb-1d67-71390ae09b88
y1 = sine_function_polynomial_1.(x1)

# ╔═╡ 4b3205e0-8084-11eb-13f1-f3a342bbc4ec
err = abs.(sin.(x1) .- y1)

# ╔═╡ Cell order:
# ╠═00615c20-807d-11eb-2e8c-e9b6bdd0fa71
# ╠═a1da079e-807d-11eb-0bcc-9975938828c4
# ╠═a1b73d60-807d-11eb-0cd2-d9ab26db9d20
# ╠═7ba588a0-807f-11eb-1bd6-f9e922d15b1d
# ╠═0429c3e0-807f-11eb-2af0-dd6ee51f5445
# ╠═a1a710c0-807d-11eb-199c-c724488ea174
# ╠═cc95df50-8082-11eb-38ac-f511ef23765f
# ╠═a190c9a2-807d-11eb-3e22-95d363159acf
# ╠═a16c0390-807d-11eb-3350-836f45affd94
# ╠═a095c7ce-807d-11eb-1700-658e8d5d5e17
# ╠═a0828df0-807d-11eb-2d94-d7deea19e939
# ╠═a06fc940-807d-11eb-0b2b-45ef5e3f3daf
# ╠═a04c14a0-807d-11eb-3a5d-e580178cd882
# ╠═c141db60-8080-11eb-27f7-bb7d4ac97a0b
# ╠═53009e80-8084-11eb-180b-616b7c6dfc3d
# ╠═32fd94d0-8084-11eb-1d67-71390ae09b88
# ╠═4b3205e0-8084-11eb-13f1-f3a342bbc4ec
