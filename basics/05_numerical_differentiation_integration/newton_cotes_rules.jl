### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ d4dfa770-7576-11eb-236c-bbc552f14db1
md"### Newton-Cotes composite rules of integrations

Chapter 5, Sauer's Numerical Analysis 
"

# ╔═╡ 5efe9070-8b2e-11eb-29fb-a3093c58fc1c
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

# ╔═╡ 5ea9e0c2-8b2e-11eb-2181-01ab4766688e
"""
Simpson's rule of integration given a function f within an interval [a,b].

	s = simpson(f::Function, a, b; rule=:te)

f:    Function
a:    Lower bound of the interval
b:    Upper bound of the interval
rule: Rule type, :ot (1/3), :te (3/8)

s: Integration value
"""
function simpson(f::Function, a, b; rule=:te)
	if rule == :ot
		s = (b - a)/6.0*(f(a) + 4.0*f((a+b)/2.0) + f(b))
	elseif rule == :te
		x = a:(b - a)/3.0:b
		s = (b - a)/8.0*(f(x[1]) + 3.0*f(x[2]) + 3.0*f(x[3]) + f(x[4]))
	end
	return s
end

# ╔═╡ 6b635300-8b2e-11eb-2f1e-e5d7f8e25feb
"""
Boole's rule of integration given a function f within an interval [a,b].
(Typographically known as Bode's rule also.)

	s = boole(f::Function, a, b)

f:    Function
a:    Lower bound of the interval
b:    Upper bound of the interval

s: Integration value
"""
function boole(f::Function, a, b)
	x = a : (b - a)/4.0 : b
	s = 0.5*(b - a)/45.0*(7.0*f(x[1]) + 32.0*f(x[2]) + 12.0*f(x[3]) + 32.0*f(x[4]) + 7.0*f(x[5]))
	return s
end

# ╔═╡ 0f4748a0-7577-11eb-2226-43ddedb96f0a
"""
Composite trapezoid rule of integration given a vector y in an interval [a,b].

	sol = composite_trapezoid(y::Vector, a, b)

y: Vector with values of the function
a: Lower bound of the interval
b: Upper bound of the interval

sol: Results
	s: Integration value
	h: Step size
"""
function composite_trapezoid(y::Vector, a, b)
	a > b ? (a, b) = (b, a) : nothing
	m = length(y) - 1
	h = (b - a) / float(m)
	s = h * ((y[1] + y[end]) / 2.0 + sum(y[2:end-1]))
	return (s=s, h=h)
end

# ╔═╡ b251c0f0-8a6e-11eb-2260-f714515d5bf2
"""
Composite trapezoid rule of integration given a function f(x) with x ∈ [a,b].

	sol = composite_trapezoid(f::Function, a, b; m=4)

f: Function
a: Lower bound of the interval
b: Upper bound of the interval
m:    Number of steps

sol: Results
	s: Integration value
	h: Step size
"""
function composite_trapezoid(f::Function, a, b; m=4)
	a > b ? (a, b) = (b, a) : nothing
	m = length(y) - 1
	h = (b - a) / float(m)
	x = a:h:b
	s = h * ((f(x[1]) + f(x[end])) / 2.0 + sum(f.(x[2:end-1])))
	
	# # Does not allocate x
	# s = (f(a) + f(b)) / 2.0
	# for i in 2:m-1
	# 	s += f(a + i*h)
	# end
	# s *= h
	
	return (s=s, h=h)
end

# ╔═╡ 0975a540-8a6f-11eb-0b8c-8d2fe16910b4
"""
Composite Simpson's rule of integration given a vector y in an interval [a,b].

	sol = composite_simpson(y::Vector, a, b; rule=:te)

y:    Vector with values of the function
a:    Lower bound of the interval
b:    Upper bound of the interval
rule: Rule type, :ot (1/3), :te (3/8)

sol: Results
	s: Integration value
	h: Step size
"""
function composite_simpson(y::Vector, a, b; rule=:te)
	a > b ? (a, b) = (b, a) : nothing
	m = length(y) - 1
	h = (b - a) / float(m)
	s = 0.0
	if rule == :ot
		rem(m, 2) == 0 || throw("length(fy must be an even integer for the 1/3 rule.")
		for i in 1:Int(m/2)
			s += y[2*i-1] + 4.0*y[2*i] + y[2*i+1] 
		end
		s *= h/3.0
	elseif rule == :te
		rem(m, 3) == 0 || throw("length(y)-1 must be an integer multiple of 3 for the 3/8 rule.")
		for i in 1:Int(m/3)
			s += y[3*i-2] + 3.0*y[3*i-1] + 3.0*y[3*i] + y[3*i+1] 
		end
		s *= 3.0*h/8.0	
	end
	return (s=s, h=h)
end

# ╔═╡ 2f610b80-8aa8-11eb-0ed5-975426478e04
"""
Composite Simpson's rule of integration given a function f within an interval [a,b].

	sol = composite_simpson(f::Function, a, b; rule=:te, m=9)

f:    Function
a:    Lower bound of the interval
b:    Upper bound of the interval
rule: Rule type, :ot (1/3), :te (3/8)
m:    Number of steps

sol: Results
	s: Integration value
	h: Step size
"""
function composite_simpson(f::Function, a, b; rule=:te, m=9)
	a > b ? (a, b) = (b, a) : nothing
	h = (b - a) / float(m)
	y = f.(a:h:b)
	s = 0.0
	if rule == :ot
		rem(m, 2) == 0 || throw("m must be an even integer for the 1/3 rule.")
		for i in 1:Int(m/2)
			s += y[2*i-1] + 4.0*y[2*i] + y[2*i+1] 
		end
		s *= h/3.0
	elseif rule == :te
		rem(m, 3) == 0 || throw("m must be an integer multiple of 3 for the 3/8 rule.")
		for i in 1:Int(m/3)
			s += y[3*i-2] + 3.0*y[3*i-1] + 3.0*y[3*i] + y[3*i+1] 
		end
		s *= 3.0*h/8.0
	end
	return (s=s, h=h)
end

# ╔═╡ aa774ef0-8b12-11eb-0698-2745d790d2e3
"""
Composite Midpoint rule of integration given a function f within an interval [a,b].

	sol = composite_midpoint(f::Function, a, b; m=10)

f:    Function
a:    Lower bound of the interval
b:    Upper bound of the interval
m:    number of steps

sol: Results
	s: Integration value
	h: Step size
"""
function composite_midpoint(f::Function, a, b; m=10)
	rem(m, 2) == 0 || throw("The length of y must be even.")
	a > b ? (a, b) = (b, a) : nothing
	h = (b - a) / m
	x = a:h:b
	x = (x[1:end-1] .+ x[2:end]) ./ 2.0 # midpoints
	s = h*sum(f.(x))
	return (s=s, h=h)
end

# ╔═╡ c6a83350-7e1f-11eb-2681-e941c89ebf30
f(x) = log(x)

# ╔═╡ c687b300-7e1f-11eb-2aaf-139e3c8baa50
a, b = 1.0, 2.0

# ╔═╡ 07a101c0-8a6e-11eb-14fc-01defc2c05fa
y = f.([1.0, 5/4, 6/4, 7/4, 2])

# ╔═╡ c671ba00-7e1f-11eb-18dd-cb0881166d53
composite_trapezoid(y, a, b)

# ╔═╡ c6618d60-7e1f-11eb-3ad1-9fd9641349f5
composite_trapezoid(f, a, b)

# ╔═╡ c64d9030-7e1f-11eb-2b52-6191b74060af
composite_simpson(y, a, b; rule=:ot)

# ╔═╡ 1e2b2160-8a72-11eb-38f5-ad0a38adff9b
x0 = a:(b-a)/6:b
composite_simpson(f.(x0), a, b)

# ╔═╡ 1d814cd0-8a72-11eb-16be-5973ba0cdb24
composite_simpson(f, a, b; m=6)

# ╔═╡ 1d6cb360-8a72-11eb-13ad-cbd719cecfb1
composite_simpson(f, a, b; rule=:ot, m=4)

# ╔═╡ 8a0270a0-8b12-11eb-1e3f-75a1a6067690
composite_midpoint(f, a, b)

# ╔═╡ 5b3654f0-8b1a-11eb-0c4e-f5308ea36a10


# ╔═╡ Cell order:
# ╠═d4dfa770-7576-11eb-236c-bbc552f14db1
# ╠═5efe9070-8b2e-11eb-29fb-a3093c58fc1c
# ╠═5ea9e0c2-8b2e-11eb-2181-01ab4766688e
# ╠═6b635300-8b2e-11eb-2f1e-e5d7f8e25feb
# ╠═0f4748a0-7577-11eb-2226-43ddedb96f0a
# ╠═b251c0f0-8a6e-11eb-2260-f714515d5bf2
# ╠═0975a540-8a6f-11eb-0b8c-8d2fe16910b4
# ╠═2f610b80-8aa8-11eb-0ed5-975426478e04
# ╠═aa774ef0-8b12-11eb-0698-2745d790d2e3
# ╠═c6a83350-7e1f-11eb-2681-e941c89ebf30
# ╠═c687b300-7e1f-11eb-2aaf-139e3c8baa50
# ╠═07a101c0-8a6e-11eb-14fc-01defc2c05fa
# ╠═c671ba00-7e1f-11eb-18dd-cb0881166d53
# ╠═c6618d60-7e1f-11eb-3ad1-9fd9641349f5
# ╠═c64d9030-7e1f-11eb-2b52-6191b74060af
# ╠═1e2b2160-8a72-11eb-38f5-ad0a38adff9b
# ╠═1d814cd0-8a72-11eb-16be-5973ba0cdb24
# ╠═1d6cb360-8a72-11eb-13ad-cbd719cecfb1
# ╠═8a0270a0-8b12-11eb-1e3f-75a1a6067690
# ╠═5b3654f0-8b1a-11eb-0c4e-f5308ea36a10
