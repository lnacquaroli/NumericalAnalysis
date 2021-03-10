### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 140f4e60-714d-11eb-3c3e-5309292dddf6
using Compat

# ╔═╡ e3cfc630-714c-11eb-1fb2-ab57b6d98aa4
md"### Li - Numerical solution to differential equations

Implementation of the upwind method to solve the 1d advection equation

$$u_t + a u_{x} = 0,\quad 0\leq x\leq 1,\quad t\geq 0,$$

$$\text{IC: } u(x,0) = \eta(x),\quad t>0$$

$$\text{BC: } u(0,t) = g(t),\quad a>0.$$
"

# ╔═╡ 0878ff60-714d-11eb-0ef4-3791db2b0299
@compat import LinearAlgebra as linalg

# ╔═╡ 0861f4ee-714d-11eb-00f2-15506da06b19
#@compat import SparseArrays as spars

# ╔═╡ 084a2730-714d-11eb-0301-61bfc046e00b
@compat import Plots as plt

# ╔═╡ 082fc15e-714d-11eb-06bb-a78e0f8560c0
md"##### Input"

# ╔═╡ 0812c380-714d-11eb-3d14-1f1f8b7825bc
domain = (a=0.0, b=1.0) # Region domain

# ╔═╡ 07fbb910-714d-11eb-1766-cd59d5feba60
nx = 20 # grid size

# ╔═╡ 07e041d0-714d-11eb-1da9-15530126fb13
t_final = 0.5 # final time

# ╔═╡ 07c6ed70-714d-11eb-23e4-99c7dce4e820
h = (domain.b - domain.a) / nx # step size

# ╔═╡ 079444b0-714d-11eb-26b0-63b3598d6f0d
Δt = h # time step

# ╔═╡ 0779b7d0-714d-11eb-3b97-1d05eff31347
a = 1.0 # wave speed

# ╔═╡ 0760b190-714d-11eb-3395-4b89905a1efa
nt = Int(round(t_final/Δt)) # total time steps

# ╔═╡ 0745fda0-714d-11eb-03c3-19417a4c9746
μ = a*Δt/h

# ╔═╡ 072cf760-714d-11eb-13ec-db8c495a6667
η(x) = x ≥ 0.5 ? 1.0 : 0.0 # initial condition

# ╔═╡ 07137bf0-714d-11eb-26a0-39cca435c229
g(t) = sin(t) # boundary condition

# ╔═╡ 06f8ef10-714d-11eb-19a6-8311178cc885
uexact(x,t) = x ≥ t ? η(x-t) : g(t-x) # exact solution

# ╔═╡ 06df9ab0-714d-11eb-1255-6d040ab8db5d
md"##### Calculations"

# ╔═╡ 06c66d60-714d-11eb-34e3-4971e54af904
function upwind(x,t,uexact,μ)
	ukm1 = uexact.(x, 0.0) # Initial condition
	
	uk = zeros(nx+1)
	u = zeros(nx+1,nt)
	for j in 1:nt
		ukm1[1] = g(t[j])
		uk[1] = g(t[j+1])
		for i in 2:nx+1
			# for a ≥ 0
			uk[i] = ukm1[i] - μ*(ukm1[i] - ukm1[i-1])
		end
		ukm1 = copy(uk)
		u[:,j] = uk
	end
	return u
end

# ╔═╡ 06ad1900-714d-11eb-20a0-49fc6b1fc6c0
x = @. domain.a + (0:nx)*h # x-axis

# ╔═╡ 0693c4a0-714d-11eb-2285-2175a7643434
t = @. (0:nt)*Δt # time-axis

# ╔═╡ 05ec881e-714d-11eb-029d-45879e159126
u = upwind(x, t, uexact, μ);

# ╔═╡ 05b01b60-714d-11eb-0516-6bbf3c5fba0b
ue = uexact.(x, t[end]);

# ╔═╡ 05ae94c0-714d-11eb-0668-8fc592be7efb
linalg.norm(ue.-u[:,end], Inf)

# ╔═╡ 05ad3530-714d-11eb-0e6d-2319941a3e1e
plt.plot(x, [ue, u[:,end], ue.-u[:,end]], layout=(1,3), title=["Analytical" "Upwind" "Error"])

# ╔═╡ 0569003e-714d-11eb-0c8c-f17049c1be43


# ╔═╡ Cell order:
# ╟─e3cfc630-714c-11eb-1fb2-ab57b6d98aa4
# ╠═140f4e60-714d-11eb-3c3e-5309292dddf6
# ╠═0878ff60-714d-11eb-0ef4-3791db2b0299
# ╠═0861f4ee-714d-11eb-00f2-15506da06b19
# ╠═084a2730-714d-11eb-0301-61bfc046e00b
# ╟─082fc15e-714d-11eb-06bb-a78e0f8560c0
# ╠═0812c380-714d-11eb-3d14-1f1f8b7825bc
# ╠═07fbb910-714d-11eb-1766-cd59d5feba60
# ╠═07e041d0-714d-11eb-1da9-15530126fb13
# ╠═07c6ed70-714d-11eb-23e4-99c7dce4e820
# ╠═079444b0-714d-11eb-26b0-63b3598d6f0d
# ╠═0779b7d0-714d-11eb-3b97-1d05eff31347
# ╠═0760b190-714d-11eb-3395-4b89905a1efa
# ╠═0745fda0-714d-11eb-03c3-19417a4c9746
# ╠═072cf760-714d-11eb-13ec-db8c495a6667
# ╠═07137bf0-714d-11eb-26a0-39cca435c229
# ╠═06f8ef10-714d-11eb-19a6-8311178cc885
# ╟─06df9ab0-714d-11eb-1255-6d040ab8db5d
# ╠═06c66d60-714d-11eb-34e3-4971e54af904
# ╠═06ad1900-714d-11eb-20a0-49fc6b1fc6c0
# ╠═0693c4a0-714d-11eb-2285-2175a7643434
# ╠═05ec881e-714d-11eb-029d-45879e159126
# ╠═05b01b60-714d-11eb-0516-6bbf3c5fba0b
# ╠═05ae94c0-714d-11eb-0668-8fc592be7efb
# ╠═05ad3530-714d-11eb-0e6d-2319941a3e1e
# ╠═0569003e-714d-11eb-0c8c-f17049c1be43
