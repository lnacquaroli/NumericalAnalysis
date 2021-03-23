### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ d4dfa770-7576-11eb-236c-bbc552f14db1
md"### Romberg integration method

Chapter 5, Sauer's Numerical Analysis 
"

# ╔═╡ 0f4748a0-7577-11eb-2226-43ddedb96f0a
"""
Computes approximation to definite integral using Romberg integration given a function in an interval [a,b].

	sol = romberg_integration(f::Function, a, b; m=10, tol=1e-4)

f:   Function to integrate
a:   Lower bound of the interval
b:   Upper bound of the interval
m:   Number of rows in the Romberg tableau
tol: Stopping criteria, when |r[j,j] - r[j+1,j+1]| ≤ tol.

sol: Results
	R:   Romberg tableau (lower-triagonal matrix)
	s:   Integration result
	err: Error = |R[end,end] - R[end-1,end-1]|
"""
function romberg_integration(f::Function, a, b; m=10, tol=1e-4)
	a > b ? (a, b) = (b, a) : nothing
	h = (b - a) ./ 2 .^(0:m-1)
	R = zeros(m, m)
	R[1,1] = 0.5*(b - a)*(f(a) + f(b))
	for j in 2:m
		s0 = 0.0
		for i in 1:2^(j-2)
			s0 += f(a + (2*i-1)*h[j])
		end
		R[j,1] = 0.5*R[j-1,1] + h[j]*s0
		for k in 2:j
			R[j,k] = (4^(k-1)*R[j,k-1] - R[j-1,k-1])/(4^(k-1) - 1)
		end
		(abs(R[j,j] - R[j-1,j-1]) ≤ tol) && return (R=R[1:j,1:j], s=R[j,j], err=abs(R[j,j] - R[j-1,j-1]))
	end
	return (R=R, s=R[end,end], err=abs(R[end-1,end-1] - R[end,end]))
end

# ╔═╡ c6a83350-7e1f-11eb-2681-e941c89ebf30
f(x) = log(x)

# ╔═╡ c687b300-7e1f-11eb-2aaf-139e3c8baa50
a, b = 1.0, 2.0

# ╔═╡ 5b3654f0-8b1a-11eb-0c4e-f5308ea36a10
romberg_integration(f, a, b)

# ╔═╡ 435e4d00-8b20-11eb-0f24-534c460d0fb5
romberg_integration(sin, a, b)

# ╔═╡ Cell order:
# ╠═d4dfa770-7576-11eb-236c-bbc552f14db1
# ╠═0f4748a0-7577-11eb-2226-43ddedb96f0a
# ╠═c6a83350-7e1f-11eb-2681-e941c89ebf30
# ╠═c687b300-7e1f-11eb-2aaf-139e3c8baa50
# ╠═5b3654f0-8b1a-11eb-0c4e-f5308ea36a10
# ╠═435e4d00-8b20-11eb-0f24-534c460d0fb5
