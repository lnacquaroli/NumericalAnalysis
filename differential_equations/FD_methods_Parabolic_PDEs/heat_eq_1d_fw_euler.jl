### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 5916f8e0-6c85-11eb-0219-2b5ec9a5b153
using Compat

# ╔═╡ 2ecbe8c0-6c9e-11eb-2eb0-2136b9826e29
using PlutoUI

# ╔═╡ 27c8bfd0-6c85-11eb-1b47-216cc3b59acf
md" ## Li - Numerical Solution to DEs

Implementation of the Forward Euler method to solve the 1d heat equation, 

$$u_t = \beta u_{xx} + f(x,t),\quad a<x<b$$

$$\text{BC1:}\, u(a,t)=g_1(t),\quad \text{BC2:}\, u(b,t)=g_2(t),\quad \text{IC:}\, u(x,0)=u_0(x).$$

We take the initial estimation as the true solution $u(x,t)=\sin(\pi x)\cos(t)$, i.e., $u_0(x)=\sin(\pi x)$, and the source term $f(x,t)=-\sin(\pi x) \sin(t) + \pi^2 \sin(\pi x) \cos(t)$.

"

# ╔═╡ 58fe8ee0-6c85-11eb-31d9-c3ecb6572363
@compat import Plots as plt

# ╔═╡ 58da16f0-6c85-11eb-2d0a-39aed7fae96b
md"##### Input"

# ╔═╡ 58be78a0-6c85-11eb-25b2-0bbbcfa2e099
a, b = 0.0, 1.0  # Region domain

# ╔═╡ 58bcf200-6c85-11eb-1b66-2f06b8f5d9da
nx, nt = 10, 20 # Grid points in x and t

# ╔═╡ 479c9140-6c87-11eb-34ee-5fb0c001543e
β = 1 # diffusivity

# ╔═╡ e875e820-6c8a-11eb-3bdc-17f367a4f5a1
f(x,t) = sin(π*x)*(π^2*cos(t) - sin(t)) # source function

# ╔═╡ eb154b20-6c8a-11eb-2a05-a506cd1154a1
uexact(x,t) = sin(π*x)*cos(t) # true solution

# ╔═╡ afcbc050-6c88-11eb-25af-395385b050d0
md"##### Calculations"

# ╔═╡ 3e7d1930-6c88-11eb-3339-b37a08a90346
""" Calculate Δt from Courant-Friedrichs-Lewy stability condition. """
function cfl(β, h)
	Δt = 0.5*h^2/β
	τ = Δt*β/h^2
	return Δt, τ
end

# ╔═╡ 4d9b0b80-6c8c-11eb-0f9c-459fba669e6e
function map_function(f, x, t)
	fs = [f(x[i],t[j]) for i in 1:length(x), j in 1:length(t)]		
	return fs
end

# ╔═╡ c6b41770-6c8a-11eb-181a-a19ed0b69591
function fw_euler(domain, nx, nt, f, uexact)
	h = (domain.x2 - domain.x1)/nx # spatial step
	Δt, τ = cfl(β, h) # time step
	
	x = @. domain.x1 + (0:nx)*h # x-axis
	t = @. (0:nt)*Δt # t-axis
	
	u = zeros(nx+1, nt+1)
	u[:,1] = map_function(uexact, x, 0.0) # initial condition
	fs = map_function(f, x, t) # source function 
	
	for j in 2:nt+1
		for i in 2:nx
			u[i,j] = u[i,j-1] + τ*(u[i-1,j-1] - 2.0*u[i,j-1] + u[i+1,j-1]) + Δt*fs[i,j-1]
		end
	end
	
	R = similar(u) # residual, local truncation error
	for i in 2:nx, j in 2:nt
		R[i,j] = u[i,j] - u[i,j-1] - τ*(u[i-1,j] + 2.0*u[i,j] + u[i+1,j]) - Δt*fs[i,j]
	end
	
	return u, x, t, R, τ
end

# ╔═╡ fcf63462-6c96-11eb-1d00-f17383c14b6a
domain = (x1=a, x2=b)

# ╔═╡ 6f28970e-6cdf-11eb-0771-9db8aae7d541
h = (domain.x2 - domain.x1)/nx

# ╔═╡ 72b3928e-6cdf-11eb-25e2-6d20a36f78cf
h^2/2

# ╔═╡ 461b4870-6c87-11eb-1f82-ab5bd26db3ba
u, x, t, R, τ = fw_euler(domain, nx, nt, f, uexact);

# ╔═╡ c50ba450-6c9a-11eb-0786-1901f6771354
ue = map_function(uexact, x, t);

# ╔═╡ fc424db0-6c96-11eb-3e77-2b5c4ac5f273
plt.plot(x, [ue[:,1], u[:,1]], line=([:path :scatter]), label=["Analytical" "FW Euler"])

# ╔═╡ fc2ace10-6c96-11eb-2542-a7f673bd2ca8
plt.contourf(x, t, [ue', u', (ue.-u)', abs.(R)'], layout=(2,2), label=["FW Euler" "Analytical" "Error" "Residual (LTE)"])

# ╔═╡ 45f067e0-6c87-11eb-0714-b1f0d0c24a35
@bind i Slider(1:length(t))

# ╔═╡ 45a18490-6c87-11eb-2bce-2936c651fba7
plt.plot(x, [ue[:,i], u[:,i]], line=([:path :scatter]), label=["Analytical" "FW Euler"])


# ╔═╡ 3e625150-6ca0-11eb-2287-039179d297f7
function li_fw_pag82()
	a, b, m, n = 0.0, 1.0, 10, 20
	h = (b-a)/m
	k = h^2/2; # Try k = h^2/1.9 to see what happens;
	
	t = 0
	tau = k/h^2
	x = zeros(m+1)
	y1 = zeros(m+1)
	y2 = zeros(m+1)
	for i=1:m+1
		x[i] = a + (i-1)*h
		y1[i] = uexact(t,x[i])
		y2[i] = 0
	end
	
	yt1=0
	yt5=0
	for j=1:n
		y1[1]=0
		y1[m+1]=0
		for i=2:m
			y2[i] = y1[i] + tau*(y1[i-1]-2*y1[i]+y1[i+1]) + k*f(t,x[i])
		end
		t = t + k
		y1 = y2
		if j==1
			yt1 = y2
		end
		if j==5
			yt5 = y2
		end
		
	end
	
	return x, y2, t, yt1, yt5
end

# ╔═╡ db2d5672-6ca4-11eb-2e44-81b5515c38fa
x2, y2, t2, yt0, yt5 = li_fw_pag82()

# ╔═╡ db100a70-6ca4-11eb-2273-cbfa68835528
plt.plot(x2,[yt0, yt5, y2])

# ╔═╡ fa34cad2-6ce0-11eb-353c-99248fd6d508
t[1]

# ╔═╡ Cell order:
# ╟─27c8bfd0-6c85-11eb-1b47-216cc3b59acf
# ╠═5916f8e0-6c85-11eb-0219-2b5ec9a5b153
# ╠═58fe8ee0-6c85-11eb-31d9-c3ecb6572363
# ╠═2ecbe8c0-6c9e-11eb-2eb0-2136b9826e29
# ╟─58da16f0-6c85-11eb-2d0a-39aed7fae96b
# ╠═58be78a0-6c85-11eb-25b2-0bbbcfa2e099
# ╠═58bcf200-6c85-11eb-1b66-2f06b8f5d9da
# ╠═479c9140-6c87-11eb-34ee-5fb0c001543e
# ╠═e875e820-6c8a-11eb-3bdc-17f367a4f5a1
# ╠═eb154b20-6c8a-11eb-2a05-a506cd1154a1
# ╟─afcbc050-6c88-11eb-25af-395385b050d0
# ╠═3e7d1930-6c88-11eb-3339-b37a08a90346
# ╠═4d9b0b80-6c8c-11eb-0f9c-459fba669e6e
# ╠═c6b41770-6c8a-11eb-181a-a19ed0b69591
# ╠═fcf63462-6c96-11eb-1d00-f17383c14b6a
# ╠═6f28970e-6cdf-11eb-0771-9db8aae7d541
# ╠═72b3928e-6cdf-11eb-25e2-6d20a36f78cf
# ╠═461b4870-6c87-11eb-1f82-ab5bd26db3ba
# ╠═c50ba450-6c9a-11eb-0786-1901f6771354
# ╠═fc424db0-6c96-11eb-3e77-2b5c4ac5f273
# ╠═fc2ace10-6c96-11eb-2542-a7f673bd2ca8
# ╠═45f067e0-6c87-11eb-0714-b1f0d0c24a35
# ╠═45a18490-6c87-11eb-2bce-2936c651fba7
# ╠═3e625150-6ca0-11eb-2287-039179d297f7
# ╠═db2d5672-6ca4-11eb-2e44-81b5515c38fa
# ╠═db100a70-6ca4-11eb-2273-cbfa68835528
# ╠═fa34cad2-6ce0-11eb-353c-99248fd6d508
