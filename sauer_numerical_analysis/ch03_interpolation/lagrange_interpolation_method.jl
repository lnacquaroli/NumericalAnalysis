### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 16fe1c92-8085-11eb-1941-fd29d589bf2a
md"### Lagrange interpolation method.

Chapter 3, Sauer's Numerical Analysis.
"

# ╔═╡ 1cc8cee2-8085-11eb-34a7-5925c626b9bb
"""
Performs Lagrange interpolation algorithm.

	y0 = lagrange_interpolation(x, y, x0)

x:  x-coordinate points
y:  y-coordinate points
x0: x-points at which interpolate

y0: y-points interpolated

Note: 
	Performs no sorting of (x,y) pairs
"""
function lagrange_interpolation(x, y, x0)
	n = length(x)
	n == length(y) || throw(DimensionMismatch("x and y must have the same length."))
	
	length(unique(x)) == n || error("x-values must be unique.")
	
	maximum(x0) ≤ maximum(x) || error("Extrapolation is not allowed.")
	minimum(x0) ≥ minimum(x) || error("Extrapolation is not allowed.")
	
	m = length(x0)
	y0 = zeros(m)
	for k in 1:m
		for i in 1:n
			u = l = 1.0
			for j in 1:n
				if j != i
					u *= x0[k] - x[j]
					l *= x[i] - x[j]
				end
			end
			y0[k] = y0[k] + u/l*y[i]
		end
	end
	
	return y0
end

# ╔═╡ 1c8695c0-8085-11eb-19b8-2da3503703c4
x = [0, 2, 3]

# ╔═╡ 1c3abfb2-8085-11eb-1f9d-cb17a043b695
y = [1, 2, 4]

# ╔═╡ 1c29a8ae-8085-11eb-0cb5-9b8e6821d438
lagrange_interpolation(x, y, [0, 2, 1.5])

# ╔═╡ c0f3fdf0-808a-11eb-388b-9fd903911342


# ╔═╡ c0c7bdd0-808a-11eb-33aa-7921c40d43d7


# ╔═╡ c0a71670-808a-11eb-0b0d-d774d7ddf6e4


# ╔═╡ c05aa422-808a-11eb-1e31-b9126bd21343


# ╔═╡ c036c870-808a-11eb-09b8-575cbb9ece1c


# ╔═╡ Cell order:
# ╟─16fe1c92-8085-11eb-1941-fd29d589bf2a
# ╠═1cc8cee2-8085-11eb-34a7-5925c626b9bb
# ╠═1c8695c0-8085-11eb-19b8-2da3503703c4
# ╠═1c3abfb2-8085-11eb-1f9d-cb17a043b695
# ╠═1c29a8ae-8085-11eb-0cb5-9b8e6821d438
# ╠═c0f3fdf0-808a-11eb-388b-9fd903911342
# ╠═c0c7bdd0-808a-11eb-33aa-7921c40d43d7
# ╠═c0a71670-808a-11eb-0b0d-d774d7ddf6e4
# ╠═c05aa422-808a-11eb-1e31-b9126bd21343
# ╠═c036c870-808a-11eb-09b8-575cbb9ece1c
