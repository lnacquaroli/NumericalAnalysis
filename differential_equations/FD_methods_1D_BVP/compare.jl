### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ a353dff0-681f-11eb-015d-75b73b19e88b
using Plots

# ╔═╡ 854337c0-681c-11eb-1498-e54c48519eb9
md"## Li - Numerical Solution of DEs

Compare the truncation errors of the forward, backward, and central scheme for approximating $ u'(x) $. We plot the error and estimate the convergence order.

We consider $ u(x) = \sin(x) $ at $ x=1 $, with exact derivative $ u'(1) = \cos(1) $.

"

# ╔═╡ a3381a90-681f-11eb-20e3-75a4a8be1943
function compare(h, m, n)
	A = zeros(m,n)
	for i = 1:m
		A[i,1] = h
		A[i,2] = (sin(1+h)-sin(1))/h - cos(1)
		A[i,3] = (sin(1) - sin(1-h))/h - cos(1)
		A[i,4] = (sin(1+h)-sin(1-h))/(2*h)- cos(1)
		h /= 2.0
	end
	return A
end

# ╔═╡ a31465f0-681f-11eb-23c9-8153b3d0c039
A0= compare(0.1, 5, 4)

# ╔═╡ a2f03c22-681f-11eb-017b-f5ab856711cd
A = abs.(A0)

# ╔═╡ a2c95330-681f-11eb-1bf3-9f94243ab2cc
h = A[:,1]

# ╔═╡ a1e97a80-681f-11eb-1cb6-5117602a5997
e1, e2, e3 = A[:,2], A[:, 3], A[:, 4]

# ╔═╡ a1ae4640-681f-11eb-299d-f5484fd4214b
plot(h, [e1, e2, e3], xaxis=:log10, yaxis=:log10, xlim=(1e-3, 1e0), ylim=(1e-6, 1e-1), label=["Forward diff" "Backward diff" "Central diff"])

# ╔═╡ 368000a0-6821-11eb-1f81-73bd087c36c0
[h  e1  e2  e3]

# ╔═╡ Cell order:
# ╠═854337c0-681c-11eb-1498-e54c48519eb9
# ╠═a353dff0-681f-11eb-015d-75b73b19e88b
# ╠═a3381a90-681f-11eb-20e3-75a4a8be1943
# ╠═a31465f0-681f-11eb-23c9-8153b3d0c039
# ╠═a2f03c22-681f-11eb-017b-f5ab856711cd
# ╠═a2c95330-681f-11eb-1bf3-9f94243ab2cc
# ╠═a1e97a80-681f-11eb-1cb6-5117602a5997
# ╠═a1ae4640-681f-11eb-299d-f5484fd4214b
# ╠═368000a0-6821-11eb-1f81-73bd087c36c0
