### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ c98e64c0-7647-11eb-0b69-8537adf60268
md"### Bisection method

Sauer's Numerical Analysis and other

"

# ╔═╡ e4eaf0d0-7647-11eb-34e7-1535103579d5
"""
Bisection Method: computes approximate solution of f(x)=0

	sol = bisection(f, a, b; ε=1.0e-6)

f:    Function,
a, b: Interval such that f(a)*f(b)<0,
ε:    Absolute error tolerance for (b-a)/2

sol:  Solution:
	xc:       Approximate solution
	iter: 	  Number of step the bisection methods run
	err:      Error upper bound to the solution > solution error = |xc - r|
	fun_eval: Number of function evaluations performed
"""
function bisection(f, a, b; ε=1.0e-6)
	sign(f(a))*sign(f(b)) < 0.0 || throw("f(a)f(b) < 0 not satisfied!")
	fa, fb = f(a), f(b)
	n = 0
	while (b - a)/2.0 > ε
		n += 1
		c = (a + b)/2.0
		fc = f(c)
		fc != 0 || return c # c is a solution, done
		if sign(fc)*sign(fa) < 0.0 # a and c make the new interval
			b, fb = c, fc
		else # c and b make the new interval
			a, fa = c, fc
		end
	end
	return (xc=(a + b)/2.0, iter=n, err=(b - a)/(2^(n+1)), fun_eval=n+2)
end

# ╔═╡ 40d99c40-764b-11eb-0d78-c9cbba9d3a48
"""
Bisection Method (version 2): computes approximate solution of f(x)=0.

	sol = bisection2(f, a, b; p=6)

f:    function,
a, b: interval such that f(a)*f(b)<0,
p:    precision [correct solution within p decimal places if the error 
      is < 0.5 × 10^(-p)]

sol:  Solution:
		xc:       Approximate solution
		iter: 	  Number of step the bisection methods run
		err:      Error upper bound to the solution > solution error = |xc - r|
		fun_eval: Number of function evaluations performed
"""
function bisection2(f, a, b; p=6)
	sign(f(a))*sign(f(b)) < 0.0 || throw("f(a)f(b) < 0 not satisfied!")
	n = ceil(log10((b - a)/(0.5*10.0^(-p)))/log10(2) - 1)
	fa, fb = f(a), f(b)
	
	for i in 1:n
		c = (a + b)/2.0
		fc = f(c)
		fc != 0 || return c # c is a solution, done
		if sign(fc)*sign(fa) < 0.0 # a and c make the new interval
			b, fb = c, fc
		else # c and b make the new interval
			a, fa = c, fc
		end
	end

	return (xc=(a + b)/2.0, iter=n, err=(b - a)/(2^(n+1)), fun_eval=n+2)
end

# ╔═╡ beceb7f0-764d-11eb-1781-71025bf1468b
"""

Bisection Method (version 3): computes approximate solution of f(x)=0, with relative error termination criteria.

	sol = bisection3(f, a, b; ε=1.0e-6)

f:    function,
a, b: interval such that f(a)*f(b)<0,
ε:    Relative error tolerance for (xi - xim1)/xi. Stopping criteria

sol:  Solution:
		xc:       Approximate solution
		iter: 	  Number of step the bisection methods run
		rel_err:  Relative error (xi - xim1)/xi
"""
function bisection3(f, xl, xu; ε=1.0e-6)
    xm = (xl + xu) / 2.0 # Calculate mid-point
    (fl, fu) = f.([xl, xu]) # Evaluate functions at boundaries
	n = 0
	δ = Inf
    while δ > ε # Convergence criteria
		n += 1
        fxm = f(xm) # Evaluate functions at mid-point
        # Adjust the bounds
        if sign(fl) == sign(fxm)
            (xl, fl) = (xm, fxm)
        else
            (xu, fu) = (xm, fxm)
        end
        # Update mid-points
        xold = xm
        xm = (xl + xu) / 2
        δ = abs((xm - xold)/xm)
    end # while
    return (xc=xm, iter=n, rel_err=δ)
end

# ╔═╡ e8730620-7647-11eb-2696-cf2f560761c2
f(x) = x^3 + x - 1.0

# ╔═╡ e8646020-7647-11eb-0b9b-573d1e449398
bisection(f, 0.0, 1.0; ε=1.0e-6)

# ╔═╡ e84edc50-7647-11eb-1b09-3bf2fd2e12fc
bisection2(f, 0.0, 1.0)

# ╔═╡ d6787530-764d-11eb-1fe3-bdbdbdcbaef5
bisection3(f, 0.0, 1.0)

# ╔═╡ e8152eb0-7647-11eb-3896-a1e23a767e24


# ╔═╡ e7ffaae0-7647-11eb-14ec-2baa23976233


# ╔═╡ e7ec49f0-7647-11eb-0c82-25ceac4ebf39


# ╔═╡ e7d71440-7647-11eb-04d6-116536ab5855


# ╔═╡ e7171fa0-7647-11eb-11cd-2393b73362e4


# ╔═╡ e6f6c65e-7647-11eb-34bd-67aff5972c3c


# ╔═╡ e6d6bb40-7647-11eb-0b38-47482573c1ad


# ╔═╡ e6b26a60-7647-11eb-3734-23e765535307


# ╔═╡ e6873bb0-7647-11eb-21e4-21f8b967e0fe


# ╔═╡ e67a6a70-7647-11eb-2e4b-2fd671c498f2


# ╔═╡ e66b7650-7647-11eb-37fa-83e92a8d1149


# ╔═╡ Cell order:
# ╟─c98e64c0-7647-11eb-0b69-8537adf60268
# ╠═e4eaf0d0-7647-11eb-34e7-1535103579d5
# ╠═40d99c40-764b-11eb-0d78-c9cbba9d3a48
# ╠═beceb7f0-764d-11eb-1781-71025bf1468b
# ╠═e8730620-7647-11eb-2696-cf2f560761c2
# ╠═e8646020-7647-11eb-0b9b-573d1e449398
# ╠═e84edc50-7647-11eb-1b09-3bf2fd2e12fc
# ╠═d6787530-764d-11eb-1fe3-bdbdbdcbaef5
# ╠═e8152eb0-7647-11eb-3896-a1e23a767e24
# ╠═e7ffaae0-7647-11eb-14ec-2baa23976233
# ╠═e7ec49f0-7647-11eb-0c82-25ceac4ebf39
# ╠═e7d71440-7647-11eb-04d6-116536ab5855
# ╠═e7171fa0-7647-11eb-11cd-2393b73362e4
# ╠═e6f6c65e-7647-11eb-34bd-67aff5972c3c
# ╠═e6d6bb40-7647-11eb-0b38-47482573c1ad
# ╠═e6b26a60-7647-11eb-3734-23e765535307
# ╠═e6873bb0-7647-11eb-21e4-21f8b967e0fe
# ╠═e67a6a70-7647-11eb-2e4b-2fd671c498f2
# ╠═e66b7650-7647-11eb-37fa-83e92a8d1149
