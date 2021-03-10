### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ e340b930-813d-11eb-049d-0fbafc60775d
using Compat

# ╔═╡ c39e460e-813d-11eb-36fe-c516f2c4860e
md"### Bezier curve interpolation method.

Chapter 3, Sauer's Numerical Analysis.
"

# ╔═╡ e6e3a980-813d-11eb-2cdd-af5a7636288d
@compat import Plots as plt

# ╔═╡ e6d74d70-813d-11eb-1a84-09eda3ffd0d0
function bezier_curves(t, r₀, r, rₚ)
	xp, yp = [], []
	m = length(r[1]) # number of points to draw
	x, y = r₀[1], r₀[2]
	for i in 1:m
		x = [x; rₚ[1][1][i]; rₚ[2][1][i]; r[1][i]]
		y = [y; rₚ[1][2][i]; rₚ[2][2][i]; r[2][i]]
		# Spline equations
		bx = 3.0*(x[2] - x[1])
		by = 3.0*(y[2] - y[1])
		cx = 3.0*(x[3] - x[2]) - bx
		cy = 3.0*(y[3] - y[2]) - by
		dx = x[4] - x[1] - bx - cx
		dy = y[4] - y[1] - by - cy
		# Horner’s method
		xp = [xp; x[1] .+ t.*(bx .+ t.*(cx .+ t.*dx))]
		yp = [yp; y[1] .+ t.*(by .+ t.*(cy .+ t.*dy))]
		x, y = x[4], y[4]		
	end
	return (x=float.(xp), y=float.(yp))
end

# ╔═╡ e6c4fdf2-813d-11eb-153a-6b733e95f48a
r0 = (1,1)

# ╔═╡ dc9312be-81b2-11eb-1850-ad35a7c17cd9
r = ([2],[2])

# ╔═╡ e6b0d9b0-813d-11eb-15cc-e1f1f8dfede5
rp = (([1], [3]), ([3], [3]))

# ╔═╡ 2768f0d0-81b3-11eb-3920-333db5bd5014
t = 0:.02:1

# ╔═╡ e550865e-813d-11eb-2f06-8fe51ef091ae
b = bezier_curves(t, r0, r, rp)

# ╔═╡ 5adbccb0-81b5-11eb-1a48-29f01cfb26f3
plt.plot(b.x, b.y, xlim=(0,4), ylim=(0,4), label="")
plt.scatter!([r0[1]], [r0[2]], label="")
plt.scatter!(r[1], r[2], label="")
plt.scatter!(rp[1][1], rp[2][1], label="")
plt.scatter!(rp[1][2], rp[2][2], label="")
plt.plot!([1, 1], [1, 3], l=:dash, label="")
plt.plot!([2, 3], [2, 3], l=:dash, label="")

# ╔═╡ 5aa26d32-81b5-11eb-3472-09decd10a7a9


# ╔═╡ 5a840fc0-81b5-11eb-130d-c369f81d84c5


# ╔═╡ 5a6f0120-81b5-11eb-2a44-95b7b4c62b32


# ╔═╡ 5a57a892-81b5-11eb-350f-ff26268276de


# ╔═╡ Cell order:
# ╟─c39e460e-813d-11eb-36fe-c516f2c4860e
# ╠═e340b930-813d-11eb-049d-0fbafc60775d
# ╠═e6e3a980-813d-11eb-2cdd-af5a7636288d
# ╠═e6d74d70-813d-11eb-1a84-09eda3ffd0d0
# ╠═e6c4fdf2-813d-11eb-153a-6b733e95f48a
# ╠═dc9312be-81b2-11eb-1850-ad35a7c17cd9
# ╠═e6b0d9b0-813d-11eb-15cc-e1f1f8dfede5
# ╠═2768f0d0-81b3-11eb-3920-333db5bd5014
# ╠═e550865e-813d-11eb-2f06-8fe51ef091ae
# ╠═5adbccb0-81b5-11eb-1a48-29f01cfb26f3
# ╠═5aa26d32-81b5-11eb-3472-09decd10a7a9
# ╠═5a840fc0-81b5-11eb-130d-c369f81d84c5
# ╠═5a6f0120-81b5-11eb-2a44-95b7b4c62b32
# ╠═5a57a892-81b5-11eb-350f-ff26268276de
