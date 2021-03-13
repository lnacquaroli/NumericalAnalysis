### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 15cc9650-7651-11eb-3abd-45791a77a120
md"### Fixed-point iteration method

Sauer's Numerical Analysis
"

# ╔═╡ 2c7b2ab0-7651-11eb-17d8-eb64c4ba32b6
"""
Fixed-Point Iteration Method
Computes approximate solution of g(x)=x

	x = fixed_point_iteration(g, x0, k)

g:  Function 
x0: Starting guess
k:  Number of iteration steps

xc: Approximate solution
"""
function fixed_point_iteration(g, x0, k)
	x1 = x0
	x2 = zero(x1)
	for i in 1:k
		x2 = g(x1)
		x1 = x2
	end
	return x2
end

# ╔═╡ 2c3a7830-7651-11eb-301e-77291347165c
g(x) = cos(x)

# ╔═╡ 2c296130-7651-11eb-1ca1-736767388658
fixed_point_iteration(g, 0.0, 10.0)

# ╔═╡ b4c4bbfe-7652-11eb-2763-5bd77e6191a1
md"Solving $x^3 + x - 1$ in three different ways with FPI."

# ╔═╡ 2b6c79d0-7651-11eb-11bd-555697875307
f1(x) = 1.0 - x^3

# ╔═╡ 20e47ba0-7653-11eb-2218-b35ebc71dc1d
fixed_point_iteration(f1, 0.5, 24)

# ╔═╡ 401fc880-7653-11eb-160c-1f512e8c88e4
fixed_point_iteration(f1, 0.5, 25)

# ╔═╡ 45294db0-7653-11eb-1389-b77fa4476cdd
fixed_point_iteration(f1, 0.5, 26)

# ╔═╡ 48c14182-7653-11eb-1094-ff1f9c2c9f01
fixed_point_iteration(f1, 0.5, 27)

# ╔═╡ 2b4c47a0-7651-11eb-145e-93b18122a6f7
f2(x) = (1.0 - x)^(1/3)

# ╔═╡ 4c50a9d0-7653-11eb-18e1-3d9ae6bf39a8
fixed_point_iteration(f2, 0.5, 25)

# ╔═╡ 2b3da1a2-7651-11eb-056f-af253f0e0017
# Add 2x^3 to both sides and solve for x
f3(x) = (1.0 + 2.0*x^3)/(1.0 + 3*x^2)

# ╔═╡ 2b24c270-7651-11eb-2793-b75912465268
fixed_point_iteration(f3, 0.5, 25)

# ╔═╡ 2af7e610-7651-11eb-05a3-0b2c696ba50f
md"Beware of the geometry of the method, as $f_1(x)$ fails to converge (spirals out) the others do not. $f_3(x)$ converges faster due to the smallest slope (smaller than one) close to the fixed point. (cobweb diagram)"

# ╔═╡ 2ac73920-7651-11eb-1ce6-e337e2bbf1c9


# ╔═╡ 2aa929d0-7651-11eb-30ac-7fb399bd0582


# ╔═╡ Cell order:
# ╟─15cc9650-7651-11eb-3abd-45791a77a120
# ╠═2c7b2ab0-7651-11eb-17d8-eb64c4ba32b6
# ╠═2c3a7830-7651-11eb-301e-77291347165c
# ╠═2c296130-7651-11eb-1ca1-736767388658
# ╠═b4c4bbfe-7652-11eb-2763-5bd77e6191a1
# ╠═2b6c79d0-7651-11eb-11bd-555697875307
# ╠═20e47ba0-7653-11eb-2218-b35ebc71dc1d
# ╠═401fc880-7653-11eb-160c-1f512e8c88e4
# ╠═45294db0-7653-11eb-1389-b77fa4476cdd
# ╠═48c14182-7653-11eb-1094-ff1f9c2c9f01
# ╠═2b4c47a0-7651-11eb-145e-93b18122a6f7
# ╠═4c50a9d0-7653-11eb-18e1-3d9ae6bf39a8
# ╠═2b3da1a2-7651-11eb-056f-af253f0e0017
# ╠═2b24c270-7651-11eb-2793-b75912465268
# ╠═2af7e610-7651-11eb-05a3-0b2c696ba50f
# ╠═2ac73920-7651-11eb-1ce6-e337e2bbf1c9
# ╠═2aa929d0-7651-11eb-30ac-7fb399bd0582
