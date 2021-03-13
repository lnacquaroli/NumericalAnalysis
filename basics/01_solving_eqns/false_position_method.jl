### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 92bbef70-7e22-11eb-3da6-8f53dc194e34
md"### False position method

[EMPossible/False Position](https://empossible.net/academics/emp4301_5301/)
"

# ╔═╡ af63c030-7e22-11eb-061a-87c268c813fc
"""
False position method for finding roots [f(x)=0].
https://empossible.net/academics/emp4301_5301/

	sol = false_position_method(f, a, b; ε=1.0e-6)

f:    Function
a, b: Interval such that f(a)*f(b)<0,
ε:    Absolute error tolerance for (xi - xim1)/xi. Stopping criteria

sol:  Solution:
	x:   Approximate solution
	err: Relative error (xi - xim1)/xi
"""
function false_position_method(f, xl, xu; ε=1.0e-6)
    (fl, fu) = f.([xl, xu]) # Evaluate functions at boundaries
    # Estimate root position
    xr = (xl + xu) / 2.0 - (xl - xu) / 2.0 * ((fu + fl)/(fu - fl))
	δ = Inf
    while δ > ε # Stopping criteria
        fxr = f(xr) # Estimate f at root position
        # Adjust the bounds
        if sign(fl) == sign(fxr)
            (xl, fl) = (xr, fxr)
        else
            (xu, fu) = (xr, fxr)
        end
        # Update root estimation
        xold = xr
        xr = (xl + xu) / 2.0 - (xl - xu) / 2.0 * ((fu + fl)/(fu - fl))
        # Calculate step between mid-points
        δ = abs((xr - xold)/xr)
    end # while
    return (x=xr, err=δ)
end

# ╔═╡ b1836be0-7e22-11eb-29f0-09c7df1f2023
f(x) = x^2 - 2

# ╔═╡ b1736650-7e22-11eb-0625-e3037a614684
false_position_method(f, 1.0, 2.0)

# ╔═╡ b1613de0-7e22-11eb-11eb-a5087ac1264e
g(x) = cos(x)

# ╔═╡ b14cf290-7e22-11eb-2ea0-e1cabd267f8f
false_position_method(g, 0.0, 0.75*π; ε=1.0e-8)

# ╔═╡ b13d3b20-7e22-11eb-3ca5-c38fd302a9a6


# ╔═╡ b122ae40-7e22-11eb-1c2d-8353bc02bc02


# ╔═╡ b0f29d90-7e22-11eb-117a-fffc12d8cdbf


# ╔═╡ b0e46cc0-7e22-11eb-09a8-c74ee6fcb4f2


# ╔═╡ b09e14f0-7e22-11eb-0f09-53cc63a89022


# ╔═╡ b089f0b0-7e22-11eb-1ee1-d5bcb12d4852


# ╔═╡ b07975f0-7e22-11eb-11a7-6b8c7b26600f


# ╔═╡ b062b9a0-7e22-11eb-18c2-95842b003d5e


# ╔═╡ b0417600-7e22-11eb-0cc6-1bba99178cf1


# ╔═╡ b02a4480-7e22-11eb-2a9e-a5dfe052863a


# ╔═╡ b01b7770-7e22-11eb-105c-57185009ceeb


# ╔═╡ b0024a22-7e22-11eb-1c34-85f716c2470c


# ╔═╡ af3fbd70-7e22-11eb-1cd0-851030e440e7


# ╔═╡ Cell order:
# ╠═92bbef70-7e22-11eb-3da6-8f53dc194e34
# ╠═af63c030-7e22-11eb-061a-87c268c813fc
# ╠═b1836be0-7e22-11eb-29f0-09c7df1f2023
# ╠═b1736650-7e22-11eb-0625-e3037a614684
# ╠═b1613de0-7e22-11eb-11eb-a5087ac1264e
# ╠═b14cf290-7e22-11eb-2ea0-e1cabd267f8f
# ╠═b13d3b20-7e22-11eb-3ca5-c38fd302a9a6
# ╠═b122ae40-7e22-11eb-1c2d-8353bc02bc02
# ╠═b0f29d90-7e22-11eb-117a-fffc12d8cdbf
# ╠═b0e46cc0-7e22-11eb-09a8-c74ee6fcb4f2
# ╠═b09e14f0-7e22-11eb-0f09-53cc63a89022
# ╠═b089f0b0-7e22-11eb-1ee1-d5bcb12d4852
# ╠═b07975f0-7e22-11eb-11a7-6b8c7b26600f
# ╠═b062b9a0-7e22-11eb-18c2-95842b003d5e
# ╠═b0417600-7e22-11eb-0cc6-1bba99178cf1
# ╠═b02a4480-7e22-11eb-2a9e-a5dfe052863a
# ╠═b01b7770-7e22-11eb-105c-57185009ceeb
# ╠═b0024a22-7e22-11eb-1c34-85f716c2470c
# ╠═af3fbd70-7e22-11eb-1cd0-851030e440e7
