### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ b7a9ff90-80f0-11eb-269d-91edca1b3c43
using Compat

# ╔═╡ 92c20520-80ef-11eb-335a-3b371d9fd22e
md"### Chebyshev interpolation method.

Chapter 3, Sauer's Numerical Analysis.
"

# ╔═╡ baefdbc0-80f0-11eb-1bf9-eb0161089587
@compat import SparseArrays as spars

# ╔═╡ ad013a52-80ef-11eb-035a-e10b93f417fc
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

# ╔═╡ acef11e0-80ef-11eb-2a39-9fe05b8c8d74
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
	n == length(y) || throw("x and y must have the same length.")
	
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

# ╔═╡ e0c8b0c0-80ef-11eb-012a-59dfbdaf9aa6
"""
Estimates the base points (knots) with the roots of the Chebyshev polynomials to interpolate the function f in the interval [a,b].

	sol = chebyshev_interpolation_nodes(a, b, f, n)

a: Minimum value of the interval
b: Maximum value of the interval
f: Function to interpolate within the interval [a,b]
n: Degree of the polynomial

sol: Solution
	x: Base points for the interpolation
	c: coefficients of the polynomials
"""
function chebyshev_interpolation_nodes(a, b, f, n)
	x = 0.5.*((b + a) .+ (b - a).*cos.((2.0.*(1:n) .- 1.0).*π./2.0./n))
	c, _ = newton_dd(x, f.(x))
	return (x=x, c=c)
end

# ╔═╡ acdbd800-80ef-11eb-121e-2dd52ff0a539
"""
Building a sin function calculator using Chebyshev interpolation polynomial.
Approximates sin curve with degree n polynomial

	y = sine_function_polynomial_2(x)

x: values to evaluate the sin at

y: approximation for sin(x)
"""
function sine_function_polynomial_2(x, c, x0)
	n = length(c)
	m = length(x)
	y = Vector{Float64}(undef, m)
	for i in 1:m
		# For each input x, move x to the fundamental domain and evaluate the
		# interpolating polynomial
		s = 1 # Correct the sign of sin
		x1 = mod(x[i], 2*π)
		if x1 > π
			x1 = 2*π - x1
			s = -1.0
		end
		if x1 > π/2
			x1 = π - x1
		end
	
		# Evaluate the polynomial
		# y = s*horner_method(n-1, c, x1; b=x0)
		y[i] = s*horner_method(n-1, c, x1; b=x0)
	end
	return y
end

# ╔═╡ acc85000-80ef-11eb-1607-95f487a21172
x1 = [1, 2, 3, 4, 14, 1000]

# ╔═╡ 0fac1970-80f2-11eb-19e6-fbdb2797befd
ci = chebyshev_interpolation_nodes(0.0, π/2, sin, 4)

# ╔═╡ 417a0d00-80f6-11eb-33b9-b53618132edc
sine_function_polynomial_2(x1, ci.c, ci.x)

# ╔═╡ ac674440-80ef-11eb-0e9b-e19c43695d68
err = abs.(sin.(x1) .- y1)

# ╔═╡ abf74460-80ef-11eb-206d-1f15f966ebfa
ci2 = chebyshev_interpolation_nodes(0.0, π/2, sin, 9)

# ╔═╡ ab4b7400-80ef-11eb-127c-cb332ea4bb94
sine_function_polynomial_2(x1, ci2.c, ci2.x)

# ╔═╡ aae6227e-80ef-11eb-3a66-35894e3e60c8


# ╔═╡ Cell order:
# ╟─92c20520-80ef-11eb-335a-3b371d9fd22e
# ╠═b7a9ff90-80f0-11eb-269d-91edca1b3c43
# ╠═baefdbc0-80f0-11eb-1bf9-eb0161089587
# ╠═ad013a52-80ef-11eb-035a-e10b93f417fc
# ╟─acef11e0-80ef-11eb-2a39-9fe05b8c8d74
# ╠═e0c8b0c0-80ef-11eb-012a-59dfbdaf9aa6
# ╠═acdbd800-80ef-11eb-121e-2dd52ff0a539
# ╠═acc85000-80ef-11eb-1607-95f487a21172
# ╠═0fac1970-80f2-11eb-19e6-fbdb2797befd
# ╠═417a0d00-80f6-11eb-33b9-b53618132edc
# ╠═ac674440-80ef-11eb-0e9b-e19c43695d68
# ╠═abf74460-80ef-11eb-206d-1f15f966ebfa
# ╠═ab4b7400-80ef-11eb-127c-cb332ea4bb94
# ╠═aae6227e-80ef-11eb-3a66-35894e3e60c8
