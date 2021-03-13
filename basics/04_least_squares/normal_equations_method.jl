### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ e340b930-813d-11eb-049d-0fbafc60775d
using Compat

# ╔═╡ c39e460e-813d-11eb-36fe-c516f2c4860e
md"### Normal equations method.

Chapter 4, Sauer's Numerical Analysis.
"

# ╔═╡ e6e3a980-813d-11eb-2cdd-af5a7636288d
# @compat import Plots as plt

# ╔═╡ af578250-81df-11eb-2d6c-697343432a73
@compat import LinearAlgebra as linalg

# ╔═╡ e6d74d70-813d-11eb-1a84-09eda3ffd0d0
function normal_equations(A, b)
	Aᵀ = transpose(A)
	aux = Aᵀ*A
	c = aux \ (Aᵀ*b)
	cn = linalg.cond(aux)
	se = sum((b .- A*c).^2)
	rmse = sqrt.(se)/length(c)
	return (c=c, cond_number=cn, rmse=rmse, se=se)
end

# ╔═╡ 5aa26d32-81b5-11eb-3472-09decd10a7a9
A = [ 1 -1; 1 0; 1 1; 1 2]

# ╔═╡ 5a840fc0-81b5-11eb-130d-c369f81d84c5
b = [1;0;0;-2]

# ╔═╡ 5a6f0120-81b5-11eb-2a44-95b7b4c62b32
sol = normal_equations(A, b)

# ╔═╡ 5a57a892-81b5-11eb-350f-ff26268276de
A2 = [ 1 -1 1; 1 0 0; 1 1 1; 1 2 4]

# ╔═╡ e4a8a600-81df-11eb-2cbf-d5940bad789e
sol2 = normal_equations(A2, b)

# ╔═╡ e48825b0-81df-11eb-2f96-5d3d44351838
x = (2 .+ (0:10) ./ 5)

# ╔═╡ e457edf0-81df-11eb-2232-77b0102f02af
y = @. 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7

# ╔═╡ e43ec0a0-81df-11eb-183b-ff5aae596d70
A3 = [x.^0 x x.^2 x.^3 x.^4 x.^5 x.^6 x.^7]

# ╔═╡ e421c2c0-81df-11eb-3d4a-b76ab08d131d
sol3 = normal_equations(A3, y)

# ╔═╡ Cell order:
# ╠═c39e460e-813d-11eb-36fe-c516f2c4860e
# ╠═e340b930-813d-11eb-049d-0fbafc60775d
# ╠═e6e3a980-813d-11eb-2cdd-af5a7636288d
# ╠═af578250-81df-11eb-2d6c-697343432a73
# ╠═e6d74d70-813d-11eb-1a84-09eda3ffd0d0
# ╠═5aa26d32-81b5-11eb-3472-09decd10a7a9
# ╠═5a840fc0-81b5-11eb-130d-c369f81d84c5
# ╠═5a6f0120-81b5-11eb-2a44-95b7b4c62b32
# ╠═5a57a892-81b5-11eb-350f-ff26268276de
# ╠═e4a8a600-81df-11eb-2cbf-d5940bad789e
# ╠═e48825b0-81df-11eb-2f96-5d3d44351838
# ╠═e457edf0-81df-11eb-2232-77b0102f02af
# ╠═e43ec0a0-81df-11eb-183b-ff5aae596d70
# ╠═e421c2c0-81df-11eb-3d4a-b76ab08d131d
