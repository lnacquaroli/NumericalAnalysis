### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ c8ee2ee0-7e23-11eb-1817-29e825b69bb3
md"### Muller method

Okten, Numerical Analysis with julia

"

# ╔═╡ e06b0fc0-7e23-11eb-152a-8f70325e5896
"""
Muller method for solving f(x) = 0.
Okten, Numerical Aanalysis with julia

	sol = muller_method(f, p; ε=1.0e-6, k=10)

f: function
p: vector with p parameters, [p0, p1, p2]
ε: Stopping criteria
k: Number of iterations

sol: Solution
	x:    approximate solution
	iter: iterations run
"""
function muller_method(f, p₀; ε=1.0e-6, k=10)
	n = 1
	q = 0
	p = Complex.(p₀)
	while n <= k
		c = f(p[3])
		b1 = (p[1] - p[3])*(f(p[2]) - f(p[3]))/((p[2] - p[3])*(p[1] - p[2]))
		b2 = (p[2] - p[3])*(f(p[1]) - f(p[3]))/((p[1] - p[3])*(p[1] - p[2]))
		b = b1 - b2
		a1 = (f(p[1]) - f(p[3]))/((p[1] - p[3])*(p[1] - p[2]))
		a2 = (f(p[2]) - f(p[3]))/((p[2] - p[3])*(p[1] - p[2]))
		a = a1 - a2
		d = (Complex(b^2 - 4.0*a*c))^0.5
		if abs(b - d) < abs(b + d)
			inc = 2.0*c/(b + d)
		else
			inc = 2.0*c/(b - d)
		end
		q = p[3] - inc
		if f(q)==0 || abs(q - p[3]) < ε
			return (p=q, y=f(q), iter=n, msg="p is $q and the iteration number is $n")
		end
		p[1] = p[2]
		p[2] = p[3]
		p[3] = q
		n += 1
	end
	y = f(q)
	return (p=q, y=y, iter=k, msg="Method did not converge. The last iteration gives $p with function value $y")
end

# ╔═╡ e051e26e-7e23-11eb-2a32-270fcfbd0786
f(x) = x^5 + 2x^3 - 5x - 2

# ╔═╡ e03ab0f0-7e23-11eb-0305-db75edb363a7
muller_method(f, [0.5, 1.0, 1.5]; ε=1e-5)

# ╔═╡ df475040-7e23-11eb-2c7c-a1fd9c04363e
muller_method(f, [0.5, 0.0, -0.1]; ε=1e-5)

# ╔═╡ df30bb00-7e23-11eb-3d1d-d354ef90d275
muller_method(f, [0.0, -0.1, -1.0]; ε=1e-5)

# ╔═╡ df167c40-7e23-11eb-0b03-d9396750e7e4
muller_method(f, [5.0, 10.0, 15.0]; ε=1e-5, k=20)

# ╔═╡ dee755f0-7e23-11eb-3684-cf2fb00d26e8


# ╔═╡ dece76c0-7e23-11eb-1bcf-017c9d12e501


# ╔═╡ Cell order:
# ╠═c8ee2ee0-7e23-11eb-1817-29e825b69bb3
# ╠═e06b0fc0-7e23-11eb-152a-8f70325e5896
# ╠═e051e26e-7e23-11eb-2a32-270fcfbd0786
# ╠═e03ab0f0-7e23-11eb-0305-db75edb363a7
# ╠═df475040-7e23-11eb-2c7c-a1fd9c04363e
# ╠═df30bb00-7e23-11eb-3d1d-d354ef90d275
# ╠═df167c40-7e23-11eb-0b03-d9396750e7e4
# ╠═dee755f0-7e23-11eb-3684-cf2fb00d26e8
# ╠═dece76c0-7e23-11eb-1bcf-017c9d12e501
