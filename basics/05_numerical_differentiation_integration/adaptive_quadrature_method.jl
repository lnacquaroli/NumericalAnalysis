### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ d4dfa770-7576-11eb-236c-bbc552f14db1
md"### Adaptive Quadrature method of integrations

Chapter 5, Sauer's Numerical Analysis 
"

# ╔═╡ b251c0f0-8a6e-11eb-2260-f714515d5bf2
"""
Trapezoid rule of integration given a function f(x) with x ∈ [a,b].

	s = trapezoid(f::Function, a, b)

f: Function
a: Lower bound of the interval
b: Upper bound of the interval

s: Integration value
"""
function trapezoid(f::Function, a, b)
	s = 0.5*(b - a)*(f(a) + f(b))
	return s
end

# ╔═╡ 2f610b80-8aa8-11eb-0ed5-975426478e04
"""
3/8 Simpson's rule of integration given a function f within an interval [a,b].

	s = simpson38(f::Function, a, b)

f:    Function
a:    Lower bound of the interval
b:    Upper bound of the interval

s: Integration value
"""
function simpson38(f::Function, a, b)
	x = LinRange(a, b, 4)
	s = (b - a)/8.0*(f(x[1]) + 3.0*(f(x[2]) + f(x[3])) + f(x[4]))
	return s
end

# ╔═╡ b9de0f70-8b29-11eb-37fa-3181efed9d82
"""
Computes approximation to definite integral of f(x) in an interval [a,b].

	sol = adaptive_quadrature_2(f, a₀, b₀, intfunc; tolfact=3.0, tol_0=5.0e-3)

f:  Function
a₀: Lower bound of the interval
b₀: Upper bound of the interval
intfunc: Function used for the integration (simpson38, trapezoid)
tolfact: Factor to scale the tolerance depending of the intfunc 
tol_0 : Tolerance for stopping criteria

sol: Results
	s: Integration value
	err: Error
"""
function adaptive_quadrature_2(f, a₀, b₀, intfunc; tolfact=10.0, tol_0=5.0e-3, maxit=1000)
	n = 1
	a = similar(zeros(maxit))
	b = similar(a)
	tol = similar(a)
	app = similar(b)
	a[1], b[1], tol[1] = a₀, b₀, tolfact*tol_0
	s = 0.0
	app[1] = intfunc(f, a[1], b[1])
	sub1 = sub2 = 0
	while n > 0 # n is current position at end of the list
		c = (a[n] + b[n]) / 2.0
		oldapp = app[n]
		app[n] = intfunc(f, a[n], c)
		app[n+1] = intfunc(f, c, b[n])
		if abs(app[n] + app[n+1] - oldapp) < tol[n] # success, done with interval
			s += (app[n] + app[n+1])
			n -= 1
			sub1 += 1
		else # set up new intervals
			sub2 += 1
			b[n+1] = b[n]
			b[n] = c
			a[n+1] = c
			tol[n] *= 0.5
			tol[n+1] = tol[n]
			n += 1 # go to end of list, repeat
		end
		(sub1 + sub2) ≤ maxit || break
	end
	return (s=s, a=a[abs.(a) .> 1e-40], b=b[abs.(b) .> 1e-40], sub1=sub1, sub2=sub2)
end

# ╔═╡ 99823520-8bde-11eb-088a-fd635419cf8b
"""
Computes recursively the approximation to definite integral of f(x) in an interval [a,b].

	s = adaptive_quadrature(f, a, b; func=simpson38, tolfact=10.0, tol=5.0e-3)

f: Function to integrate
a: Lower bound of the interval
b: Upper bound of the interval
func: Function used for the integration (simpson38, trapezoid)
tolfact: Factor to scale the tolerance depending of the intfunc (for trapezoid is 3.0)
tol : Tolerance for stopping criteria

s: Integration value

ref: https://en.wikipedia.org/wiki/Adaptive_Simpson%27s_method#Python
"""
function adaptive_quadrature(f, a, b; func=simpson38, tol=5e-3, tolfact=10.0)
    s = func(f, a, b)
    return _recursive_adaptive_quadrature(f, a, b, tol, s, func, tolfact)
end

# ╔═╡ cf518cb0-8be2-11eb-20bb-d3023e357a3a
function _recursive_adaptive_quadrature(f, a, b, tol, s, func, tolfact)
    c = (a + b)/2.0
    sl = func(f, a, c)
    sr = func(f, c, b)
    δ = sl + sr - s
	(abs(δ) ≤ tolfact*tol) && return sl + sr + δ/tolfact
    return _recursive_adaptive_quadrature(f, a, c, tol/2.0, sl, func, tolfact) + _recursive_adaptive_quadrature(f, c, b, tol/2.0, sr, func, tolfact)
end

# ╔═╡ c6a83350-7e1f-11eb-2681-e941c89ebf30
f(x) = 1 + sin(exp(3.0*x))

# ╔═╡ c687b300-7e1f-11eb-2aaf-139e3c8baa50
a, b = -1.0, 1.0

# ╔═╡ 5b3654f0-8b1a-11eb-0c4e-f5308ea36a10
trapezoid(f, a, b)

# ╔═╡ 58269b90-8b28-11eb-2768-6ba68eab43b9
simpson38(f, a, b)

# ╔═╡ d420b0ee-8b32-11eb-2ab3-198a31837889
adaptive_quadrature_2(f, a, b, simpson38; maxit=100, tolfact=15)

# ╔═╡ 57d435d0-8b28-11eb-3237-a1d8bcde3620
adaptive_quadrature_2(f, a, b, trapezoid; maxit=150, tolfact=3)

# ╔═╡ 4e59f290-8b34-11eb-1a2a-63522bc7f910
adaptive_quadrature(f, a, b; func=trapezoid, tol=5e-5, tolfact=3)

# ╔═╡ 4e477c00-8b34-11eb-0033-2b468db644d2
adaptive_quadrature(f, a, b; tol=5e-5)

# ╔═╡ 4e19b540-8b34-11eb-2250-b35ae5477656


# ╔═╡ 4dff0150-8b34-11eb-2858-4ff64d041b96


# ╔═╡ 4de75aa0-8b34-11eb-25ac-31241cd94f1e


# ╔═╡ 4dcbe360-8b34-11eb-1b72-53a4d6c3daf1


# ╔═╡ Cell order:
# ╟─d4dfa770-7576-11eb-236c-bbc552f14db1
# ╠═b251c0f0-8a6e-11eb-2260-f714515d5bf2
# ╠═2f610b80-8aa8-11eb-0ed5-975426478e04
# ╠═b9de0f70-8b29-11eb-37fa-3181efed9d82
# ╠═99823520-8bde-11eb-088a-fd635419cf8b
# ╠═cf518cb0-8be2-11eb-20bb-d3023e357a3a
# ╠═c6a83350-7e1f-11eb-2681-e941c89ebf30
# ╠═c687b300-7e1f-11eb-2aaf-139e3c8baa50
# ╠═5b3654f0-8b1a-11eb-0c4e-f5308ea36a10
# ╠═58269b90-8b28-11eb-2768-6ba68eab43b9
# ╠═d420b0ee-8b32-11eb-2ab3-198a31837889
# ╠═57d435d0-8b28-11eb-3237-a1d8bcde3620
# ╠═4e59f290-8b34-11eb-1a2a-63522bc7f910
# ╠═4e477c00-8b34-11eb-0033-2b468db644d2
# ╠═4e19b540-8b34-11eb-2250-b35ae5477656
# ╠═4dff0150-8b34-11eb-2858-4ff64d041b96
# ╠═4de75aa0-8b34-11eb-25ac-31241cd94f1e
# ╠═4dcbe360-8b34-11eb-1b72-53a4d6c3daf1
