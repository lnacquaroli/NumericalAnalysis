### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 57446a50-713c-11eb-03c4-8d755708e769
using Compat

# ╔═╡ bdef2e30-713b-11eb-3161-db3985481257
md"### Li - Numerical solution to differential equations

Implementation of the Lax-Friedrichs method to solve the 1d advection equation

$$u_t + a u_{x} = 0,\quad 0\leq x\leq 1,\quad t\geq 0,$$

$$\text{IC: } u(x,0) = \eta(x),\quad t>0$$

$$\text{BC: } u(0,t) = g(t),\quad a>0.$$
"

# ╔═╡ 57321ad0-713c-11eb-12b5-e7579cfce92e
@compat import LinearAlgebra as linalg

# ╔═╡ 57241110-713c-11eb-0216-193ede7c3527
#@compat import SparseArrays as spars

# ╔═╡ 57098430-713c-11eb-007c-112b802f034c
@compat import Plots as plt

# ╔═╡ 56af7d50-713c-11eb-14f8-237a54a36b0c
md"##### Input"

# ╔═╡ 569e6650-713c-11eb-2290-03cf941967a9
domain = (a=0.0, b=1.0) # Region domain

# ╔═╡ 5688e280-713c-11eb-12f1-6951d5be4fc8
nx = 20 # grid size

# ╔═╡ 56724d40-713c-11eb-07f8-3dd90b5da5a4
t_final = 0.5 # final time

# ╔═╡ 564263a0-713c-11eb-16c0-6520a6c5f9eb
h = (domain.b - domain.a) / nx # step size

# ╔═╡ 5629f9a0-713c-11eb-10b0-255c490dd6c4
Δt = h # time step

# ╔═╡ b6c0d7c0-713c-11eb-0061-a1d25494eed7
a = 1.0 # wave speed

# ╔═╡ 36b56260-713e-11eb-3632-1323f57b513c
nt = Int(round(t_final/Δt)) # total time steps

# ╔═╡ 55f88962-713c-11eb-10a5-e168ace6b7ad
μ = a*Δt/h/2.0

# ╔═╡ 0dfb032e-713d-11eb-0221-25577d49e57f
η(x) = x ≥ 0.5 ? 1.0 : 0.0 # initial condition

# ╔═╡ f78bf360-713d-11eb-3db5-f5f212234137
g(t) = sin(t) # boundary condition

# ╔═╡ 55de2390-713c-11eb-2126-455fa4e7e99d
uexact(x,t) = x ≥ t ? η(x-t) : g(t-x) # exact solution

# ╔═╡ 659c1150-713e-11eb-20a2-afba47e3bd13
md"##### Calculations"

# ╔═╡ 55a141a0-713c-11eb-3eb7-89dfbb812e74
function lax_friedrichs(x,t,uexact,μ)
	ukm1 = uexact.(x, 0.0) # Initial condition
	
	uk = zeros(nx+1)
	u = zeros(nx+1,nt)
	for j in 1:nt
		ukm1[1] = g(t[j])
		uk[1] = g(t[j+1])
		for i in 2:nx
			uk[i] = 0.5*(ukm1[i-1] + ukm1[i+1]) - μ*(ukm1[i+1] - ukm1[i-1])
		end
		uk[end] = ukm1[end] - μ*(ukm1[end] - ukm1[end-1])
		ukm1 = copy(uk)
		u[:,j] = uk
	end
	return u
end

# ╔═╡ 55826f00-713c-11eb-1990-8bbda1923184
x = @. domain.a + (0:nx)*h # x-axis

# ╔═╡ 556dfc9e-713c-11eb-3223-0ba2c14931cf
t = @. (0:nt)*Δt # time-axis

# ╔═╡ 55141cd0-713c-11eb-03af-2b52178bab74
u = lax_friedrichs(x, t, uexact, μ);

# ╔═╡ b0e49c70-7140-11eb-1003-3b721f9c9353
ue = uexact.(x, t[end]);

# ╔═╡ 54ff5c50-713c-11eb-1a28-d9a7fb8c5526
linalg.norm(ue.-u[:,end], Inf)

# ╔═╡ 54e9d880-713c-11eb-1751-37e4f7d2e932
plt.plot(x, [ue, u[:,end], ue.-u[:,end]], layout=(1,3), title=["Analytical" "LF" "Error"])

# ╔═╡ 5470b0e0-713c-11eb-34dc-13722f2be197


# ╔═╡ 536f6d80-713c-11eb-135b-bdd3bf4f6846


# ╔═╡ 536e5c10-713c-11eb-2643-93319f7cf134


# ╔═╡ 536cfc7e-713c-11eb-3b39-65b1e65096c4


# ╔═╡ 5325ba4e-713c-11eb-0486-e7d065581cdd


# ╔═╡ Cell order:
# ╟─bdef2e30-713b-11eb-3161-db3985481257
# ╠═57446a50-713c-11eb-03c4-8d755708e769
# ╠═57321ad0-713c-11eb-12b5-e7579cfce92e
# ╠═57241110-713c-11eb-0216-193ede7c3527
# ╠═57098430-713c-11eb-007c-112b802f034c
# ╟─56af7d50-713c-11eb-14f8-237a54a36b0c
# ╠═569e6650-713c-11eb-2290-03cf941967a9
# ╠═5688e280-713c-11eb-12f1-6951d5be4fc8
# ╠═56724d40-713c-11eb-07f8-3dd90b5da5a4
# ╠═564263a0-713c-11eb-16c0-6520a6c5f9eb
# ╠═5629f9a0-713c-11eb-10b0-255c490dd6c4
# ╠═b6c0d7c0-713c-11eb-0061-a1d25494eed7
# ╠═36b56260-713e-11eb-3632-1323f57b513c
# ╠═55f88962-713c-11eb-10a5-e168ace6b7ad
# ╠═0dfb032e-713d-11eb-0221-25577d49e57f
# ╠═f78bf360-713d-11eb-3db5-f5f212234137
# ╠═55de2390-713c-11eb-2126-455fa4e7e99d
# ╟─659c1150-713e-11eb-20a2-afba47e3bd13
# ╠═55a141a0-713c-11eb-3eb7-89dfbb812e74
# ╠═55826f00-713c-11eb-1990-8bbda1923184
# ╠═556dfc9e-713c-11eb-3223-0ba2c14931cf
# ╠═55141cd0-713c-11eb-03af-2b52178bab74
# ╠═b0e49c70-7140-11eb-1003-3b721f9c9353
# ╠═54ff5c50-713c-11eb-1a28-d9a7fb8c5526
# ╠═54e9d880-713c-11eb-1751-37e4f7d2e932
# ╠═5470b0e0-713c-11eb-34dc-13722f2be197
# ╠═536f6d80-713c-11eb-135b-bdd3bf4f6846
# ╠═536e5c10-713c-11eb-2643-93319f7cf134
# ╠═536cfc7e-713c-11eb-3b39-65b1e65096c4
# ╠═5325ba4e-713c-11eb-0486-e7d065581cdd
