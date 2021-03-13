### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ c8ee2ee0-7e23-11eb-1817-29e825b69bb3
md"### Secant method

[EMPossible/Secant Method](https://empossible.net/academics/emp4301_5301/)
"

# ╔═╡ e06b0fc0-7e23-11eb-152a-8f70325e5896
"""
Secant method for finding roots [f(x)=0].
https://empossible.net/academics/emp4301_5301/

	sol = secant_method(f, x1, x2; ε=1.0e-6)

f:      Function
x1, x2: Starting guess
ε:      Error tolerance for f[x2]/(f[x2] - f[x1])*(x2 - x1). Stopping criteria

sol:  Solution:
	x:    Approximate solution
	err:  Relative error (xi - xim1)/xi
	fval: Function value at x 
"""
function secant_method(f, x1, x2; ε=1.0e-6)
    f1 = f(x1) # Evaluate function at x1
	δ = Inf
    while abs(δ) > ε # Stopping criteria
        f2 = f(x2)
        δ = f2 / (f2 - f1) * (x2 - x1) # Calculate error
        x1, f1 = x2, f2 # Update point 1 and f1
        x2 -= δ # Update point 2
    end
    return (x=x2, err=δ, fval=f(x2))
end

# ╔═╡ e051e26e-7e23-11eb-2a32-270fcfbd0786
f(x) = x^2 - 2

# ╔═╡ e03ab0f0-7e23-11eb-0305-db75edb363a7
secant_method(f, 1.0, 2.5)

# ╔═╡ e022bc22-7e23-11eb-0ee5-c17face27bec
g(x) = cos(x)

# ╔═╡ e0063370-7e23-11eb-011b-dd67b6e5a69e
secant_method(g, 1.0, 2.5)

# ╔═╡ df5e0c90-7e23-11eb-35ee-29e416630bea


# ╔═╡ df475040-7e23-11eb-2c7c-a1fd9c04363e


# ╔═╡ df30bb00-7e23-11eb-3d1d-d354ef90d275


# ╔═╡ df167c40-7e23-11eb-0b03-d9396750e7e4


# ╔═╡ deff4ac0-7e23-11eb-31c1-61fcf3efc7d3


# ╔═╡ dee755f0-7e23-11eb-3684-cf2fb00d26e8


# ╔═╡ dece76c0-7e23-11eb-1bcf-017c9d12e501


# ╔═╡ Cell order:
# ╠═c8ee2ee0-7e23-11eb-1817-29e825b69bb3
# ╠═e06b0fc0-7e23-11eb-152a-8f70325e5896
# ╠═e051e26e-7e23-11eb-2a32-270fcfbd0786
# ╠═e03ab0f0-7e23-11eb-0305-db75edb363a7
# ╠═e022bc22-7e23-11eb-0ee5-c17face27bec
# ╠═e0063370-7e23-11eb-011b-dd67b6e5a69e
# ╠═df5e0c90-7e23-11eb-35ee-29e416630bea
# ╠═df475040-7e23-11eb-2c7c-a1fd9c04363e
# ╠═df30bb00-7e23-11eb-3d1d-d354ef90d275
# ╠═df167c40-7e23-11eb-0b03-d9396750e7e4
# ╠═deff4ac0-7e23-11eb-31c1-61fcf3efc7d3
# ╠═dee755f0-7e23-11eb-3684-cf2fb00d26e8
# ╠═dece76c0-7e23-11eb-1bcf-017c9d12e501
