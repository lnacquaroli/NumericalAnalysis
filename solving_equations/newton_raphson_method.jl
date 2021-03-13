### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ f241c0d0-7e24-11eb-2031-697fc56be16b
md"### Newton-Raphson method

[EMPossible/Newton-Raphson](https://empossible.net/academics/emp4301_5301/)
"

# ╔═╡ 138fc110-7e25-11eb-1e12-c1e450aca2d9
"""
Newton-Raphson method for finding root [f(x)=0].
https://empossible.net/academics/emp4301_5301/

	sol = newton_raphson_method(f, fp, x₀; ε=1.0e-6)

f:  Function
fp: Derivative of the function f
x₀: Initial guess
ε:  Tolerance for f(x)/fp(x). Stopping criteria

sol: Solution:
	x:    Root
	err:  Error
	fval: Function value at x
"""
function newton_raphson_method(f, fp, x₀; ε=1.0e-6)
    δ = f(x₀) / fp(x₀)
    while abs(δ) > ε # Convergence criteria
        x₀ -= δ # Update root estimate
        δ = f(x₀) / fp(x₀)
    end # while
    return (x=x₀, err=δ, fval=f(x₀))
end

# ╔═╡ 1364b970-7e25-11eb-2c14-f3bea1e5a87a
f(x) = x^2 - 2

# ╔═╡ 1349b762-7e25-11eb-036d-3564afa80341
fp(x) = 2*x

# ╔═╡ 133210ae-7e25-11eb-0948-bfdd41b381ee
newton_raphson_method(f, fp, 1.20)

# ╔═╡ 131a1be0-7e25-11eb-0124-dfaf38e0009e
g(x) = cos(x)

# ╔═╡ 12ff40e0-7e25-11eb-33af-41f45d32659c
gp(x) = -sin(x)

# ╔═╡ 12281ac2-7e25-11eb-38e3-61eb702d7004
newton_raphson_method(g, gp, 1.20)

# ╔═╡ 12104d00-7e25-11eb-020f-ad6a774415a8


# ╔═╡ 11f68370-7e25-11eb-314c-13f52a50c6d5


# ╔═╡ 11dc6bc0-7e25-11eb-1dae-ff049b47ff6b


# ╔═╡ 11c672c0-7e25-11eb-3842-8d1cdc2177ce


# ╔═╡ 11ad6c80-7e25-11eb-13d3-a30e023d682a


# ╔═╡ 11921c50-7e25-11eb-042e-0bfba7280046


# ╔═╡ Cell order:
# ╠═f241c0d0-7e24-11eb-2031-697fc56be16b
# ╠═138fc110-7e25-11eb-1e12-c1e450aca2d9
# ╠═1364b970-7e25-11eb-2c14-f3bea1e5a87a
# ╠═1349b762-7e25-11eb-036d-3564afa80341
# ╠═133210ae-7e25-11eb-0948-bfdd41b381ee
# ╠═131a1be0-7e25-11eb-0124-dfaf38e0009e
# ╠═12ff40e0-7e25-11eb-33af-41f45d32659c
# ╠═12281ac2-7e25-11eb-38e3-61eb702d7004
# ╠═12104d00-7e25-11eb-020f-ad6a774415a8
# ╠═11f68370-7e25-11eb-314c-13f52a50c6d5
# ╠═11dc6bc0-7e25-11eb-1dae-ff049b47ff6b
# ╠═11c672c0-7e25-11eb-3842-8d1cdc2177ce
# ╠═11ad6c80-7e25-11eb-13d3-a30e023d682a
# ╠═11921c50-7e25-11eb-042e-0bfba7280046
