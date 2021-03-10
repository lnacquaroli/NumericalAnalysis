### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ d4dfa770-7576-11eb-236c-bbc552f14db1
md"### Horner's method for nested multiplication

Sauer's Numerical Analysis 
"

# ╔═╡ 0f4748a0-7577-11eb-2226-43ddedb96f0a
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

# ╔═╡ 0f382d70-7577-11eb-152e-9be0bde70408
horner_method(4, [-1.0 5.0 -3.0 3.0 2.0], 1/2)

# ╔═╡ 0f1ed910-7577-11eb-19a7-13cff9783756
horner_method(4, [-1.0 5.0 -3.0 3.0 2.0], [-2.0 -1.0 0.0 1.0 2.0])

# ╔═╡ 0efecdf0-7577-11eb-28e3-992622052fcf
horner_method(3, [1.0 1/2 1/2 -1/2], 1.0; b=[0.0 2.0 3.0])

# ╔═╡ c6a83350-7e1f-11eb-2681-e941c89ebf30


# ╔═╡ c687b300-7e1f-11eb-2aaf-139e3c8baa50


# ╔═╡ c671ba00-7e1f-11eb-18dd-cb0881166d53


# ╔═╡ c6618d60-7e1f-11eb-3ad1-9fd9641349f5


# ╔═╡ c64d9030-7e1f-11eb-2b52-6191b74060af


# ╔═╡ c6391dd0-7e1f-11eb-0426-6bc31f2cb2ba


# ╔═╡ c5f50ff0-7e1f-11eb-0719-1fcfb47d3b75


# ╔═╡ Cell order:
# ╟─d4dfa770-7576-11eb-236c-bbc552f14db1
# ╠═0f4748a0-7577-11eb-2226-43ddedb96f0a
# ╠═0f382d70-7577-11eb-152e-9be0bde70408
# ╠═0f1ed910-7577-11eb-19a7-13cff9783756
# ╠═0efecdf0-7577-11eb-28e3-992622052fcf
# ╠═c6a83350-7e1f-11eb-2681-e941c89ebf30
# ╠═c687b300-7e1f-11eb-2aaf-139e3c8baa50
# ╠═c671ba00-7e1f-11eb-18dd-cb0881166d53
# ╠═c6618d60-7e1f-11eb-3ad1-9fd9641349f5
# ╠═c64d9030-7e1f-11eb-2b52-6191b74060af
# ╠═c6391dd0-7e1f-11eb-0426-6bc31f2cb2ba
# ╠═c5f50ff0-7e1f-11eb-0719-1fcfb47d3b75
