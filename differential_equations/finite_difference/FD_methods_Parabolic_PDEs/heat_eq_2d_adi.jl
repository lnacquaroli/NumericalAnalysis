### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ f846cb60-6f2e-11eb-2465-4981c2a5a085
using Compat

# ╔═╡ 19de7440-6f2e-11eb-31c9-85f019d5575a
md"### Li - Numerical solution to differential equations

Implementation of the ADI method to solve the 2d heat equation

$$u_y = u_{xx}+u_{yy} + f(x,y,t),\quad a\leq x\leq b,\quad c\leq y\leq d,\quad t\geq 0,$$

with $f(x,y,t) = \exp(-t) \sin(\pi x) \sin(\pi y) (2\pi^2-1) $ and using the exact solution, $u(x,y,t) = \exp(-t) \sin(\pi x) \sin(\pi y)$, in the boundary and initial conditions.
"

# ╔═╡ f819eefe-6f2e-11eb-1a9e-4131d38dca4f
@compat import LinearAlgebra as linalg

# ╔═╡ f7a81a60-6f2e-11eb-39f5-9727c1bb155a
@compat import SparseArrays as spars

# ╔═╡ f7918520-6f2e-11eb-3af1-03973e57e215
@compat import Plots as plt

# ╔═╡ f318801e-6f2e-11eb-1ee1-0165982824e6
md"##### Input"

# ╔═╡ f3085380-6f2e-11eb-2dbc-6566d6751265
domain = (a=0.0, b=1.0, c=0.0, d=1.0) # Region domain 

# ╔═╡ f2f1e552-6f2e-11eb-2585-073e88af63d7
n = 40 # grid size n = nx = ny

# ╔═╡ f2dd9a00-6f2e-11eb-3e5a-f56789b381f9
t_final = 0.5 # final time

# ╔═╡ a0e23f20-6f2f-11eb-2fd7-a949d5eca41c
h = (domain.b - domain.a) / n # step size for nx and ny

# ╔═╡ 9398e8f0-6f2f-11eb-0d23-054a8b5fb679
Δt = h # time step

# ╔═╡ 9388e360-6f2f-11eb-34b5-351590e9a0c7
uexact(x,y,t) = exp(-t)*sin(π*x)*sin(π*y) # exact solution

# ╔═╡ 936afb20-6f2f-11eb-2296-3d25244b1f8e
f(x,y,t) = (2.0*π^2-1.0)*exp(-t)*sin(π*x)*sin(π*y) # source term

# ╔═╡ 9357e850-6f2f-11eb-2d9d-5d7f91b18c7b
md"##### Calculations"

# ╔═╡ 933a0012-6f2f-11eb-3035-a752ff462cc5
function map_function(f, x, y, t)
	fs = [f(x[i],y[j],t[k]) for i=1:length(x), j=1:length(y), k=1:length(t)]
	return fs
end

# ╔═╡ 891db4a0-6f34-11eb-0664-01d1220073c7
function boundary_conditions(f, t, x, y)
	u = spars.spzeros(length(x), length(y))
	# x-BC
	u[:,1] = map_function(f, x, y[1], t)
	u[:,end] = map_function(f, x, y[end], t)
	# y-BC
	u[1,:] = map_function(f, x[1], y, t)
	u[end,:] = map_function(f, x[end], y, t)
	return u
end

# ╔═╡ f6f42f80-6f35-11eb-1216-49c150378a11
function loop_x_direction!(u2, u1, uexact, x, nx, y, ny, t1, t2, h, f, Δt)
	hhinv = 1.0/(h*h)
	
	for j in 2:ny # y-direction
		A = spars.spzeros(nx-1, ny-1)
		b = zeros(ny-1)
		
		for i in 2:nx # x-direction
			b[i-1] = (u1[i,j-1] - 2.0*u1[i,j] + u1[i,j+1])*hhinv + f(x[i],y[j],t2) + 2.0*u1[i,j]/Δt
			if i==2
				b[i-1] = b[i-1] + uexact(x[i-1],y[j],t2)*hhinv
				A[i-1,i] = -hhinv
			elseif i==nx
				b[i-1] = b[i-1] + uexact(x[i+1],y[j],t2)*hhinv
				A[i-1,i-2] = -hhinv
			else
				A[i-1,i] = -hhinv
				A[i-1,i-2] = -hhinv
			end
			A[i-1,i-1] = 2.0/Δt + 2.0*hhinv
		end
		ut = A \ b # Solve the diagonal matrix
		for i in 1:nx-1
			u2[i+1,j] = ut[i]
		end
	end # y-direction
end

# ╔═╡ d9beed60-6f99-11eb-3df0-0520d4a4ab01
function loop_y_direction!(u1, u2, uexact, x, nx, y, ny, t1, t2, h, f, Δt)
	hhinv = 1.0/(h*h)
	
	for i in 2:nx # y-direction
		A = spars.spzeros(nx-1, ny-1)
		b = zeros(ny-1)
		
		for j in 2:ny # x-direction
			b[j-1] = (u2[i-1,j] - 2.0*u2[i,j] + u2[i+1,j])*hhinv + f(x[i],y[j],t2) + 2.0*u2[i,j]/Δt
			if j==2
				b[j-1] = b[j-1] + uexact(x[i],y[j-1],t1)*hhinv
				A[j-1,j] = -hhinv
			elseif j==ny
				b[j-1] = b[j-1] + uexact(x[i],y[j+1],t1)*hhinv
				A[j-1,j-2] = -hhinv
			else
				A[j-1,j] = -hhinv
				A[j-1,j-2] = -hhinv
			end
			A[j-1,j-1] = 2.0/Δt + 2.0*hhinv
		end
		ut = A \ b # Solve the diagonal matrix
		for j in 1:ny-1
			u1[i,j+1] = ut[j]
		end
	end # y-direction
end

# ╔═╡ 9319cde0-6f2f-11eb-118d-79b506aac9b6
x, y = domain.a:h:domain.b, domain.c:h:domain.d # x- and y-axis

# ╔═╡ 92f46b90-6f2f-11eb-0463-43567de8cef4
ui = map_function(uexact, x, y, 0.0)[:,:]; # initial condition

# ╔═╡ f2c72bd0-6f2e-11eb-314f-1d1b3f74e1c2
nt = Int(round(t_final/Δt)) # number of time steps

# ╔═╡ f2b09690-6f2e-11eb-0887-791d4fe4a43c
t = (0:nt)*Δt # time axis

# ╔═╡ f27e14e0-6f2e-11eb-373a-5ff0bdd189a4
function adi_2d(nx, x, ny, y, nt, t, uexact, f, ui, h, Δt)
	
	u2 = Array{Float64}(undef,nx+1,ny+1)
	u1 = copy(ui)
	
	for k = 1:nt # time loop
		t1 = t[k] + Δt
		t2 = t[k] + Δt/2.0
		
		# x-sweep
		u2 = boundary_conditions(uexact, t2, x, y)
		loop_x_direction!(u2, u1, uexact, x, nx, y, ny, t2, t2, h, f, Δt)

		# y-sweep
		u1 = boundary_conditions(uexact, t1, x, y)
		loop_y_direction!(u1, u2, uexact, x, nx, y, ny, t1, t2, h, f, Δt)
	end
	
	return u1, u2
end

# ╔═╡ c602b840-6f3c-11eb-3749-c17f51dedfff
u1, u2 = adi_2d(n, x, n, y, nt, t, uexact, f, ui, h, Δt);

# ╔═╡ f2142f80-6f2e-11eb-0d00-b169cc121d49
ue = map_function(uexact, x, y, t_final)[:,:]; # true solution at final time 

# ╔═╡ f1fe84a0-6f2e-11eb-0436-2765003fd976
err = linalg.norm(u1.-ue, Inf) # The infinity error

# ╔═╡ eecea170-6f2e-11eb-33ad-314c0dd4bb86
plt.contourf(x, y, [ue', u1', (u1.-ue)'], layout=(1,3), title=["Analytical" "ADI" "Error"])

# ╔═╡ d855f970-6f97-11eb-3e40-6d89a03fb452
function li_adi(uexact, f)
	a, b, c, d, n, tfinal = 0.0, 1.0, 0.0, 1.0, 40, 0.5
	m = n
	h = (b-a)/n
	dt = h
	h1 = h*h
	x = a:h:b
	y = c:h:d
	
	# Initial condition
	t = 0
	u1 = zeros(m+1,m+1)
	for i=1:m+1
		for j=1:m+1,
			u1[i,j] = uexact(x[i],y[j],t)
		end
	end
	
	# Big loop for time t
	k_t = round(tfinal/dt)
	u2 = similar(u1)
	A = 0.0
	for k=1:k_t
		t1 = t + dt
		t2 = t + dt/2
		
		# sweep in x-direction
		
		for i=1:m+1 # Boundary condition.
			u2[i,1] = uexact(x[i],y[1],t2)
			u2[i,n+1] = uexact(x[i],y[n+1],t2)
			u2[1,i] = uexact(x[1],y[i],t2)
			u2[m+1,i] = uexact(x[m+1],y[i],t2)
		end
		
		for j = 2:n # Look for fixed y(j) 
			A = spars.spzeros(m-1,m-1)
			b = zeros(m-1)
			for i=2:m
				b[i-1] = (u1[i,j-1] - 2.0*u1[i,j] + u1[i,j+1])/h1 + f(x[i],y[j],t2) + 2.0*u1[i,j]/dt
				if i == 2
					b[i-1] = b[i-1] + uexact(x[i-1],y[j],t2)/h1
					A[i-1,i] = -1.0/h1
				else
					if i==m
						b[i-1] = b[i-1] + uexact(x[i+1],y[j],t2)/h1
						A[i-1,i-2] = -1.0/h1
					else
						A[i-1,i] = -1.0/h1
						A[i-1,i-2] = -1.0/h1
					end
				end
				A[i-1,i-1] = 2.0/dt + 2.0/h1
			end
			ut = A\b # Solve the diagonal matrix.
			for i=1:m-1
				u2[i+1,j] = ut[i]
			end
		end # Finish x-sweep.
		
		# sweep in y-direction
		
		for i=1:m+1 # Boundary condition.
			u1[i,1] = uexact(x[i],y[1],t1)
			u1[i,n+1] = uexact(x[i],y[n+1],t1)
			u1[1,i] = uexact(x[1],y[i],t1)
			u1[m+1,i] = uexact(x[m+1],y[i],t1)
		end
		
		for i = 2:m # Look for fixed x(j) 
			A = spars.spzeros(m-1,m-1)
			b = zeros(m-1)
			for j=2:n
				b[j-1] = (u2[i-1,j] - 2.0*u2[i,j] + u2[i+1,j])/h1 + f(x[i],y[j],t2) + 2.0*u2[i,j]/dt
				if j == 2
					b[j-1] = b[j-1] + uexact(x[i],y[j-1],t1)/h1
					A[j-1,j] = -1.0/h1
				else
					if j==n
						b[j-1] = b[j-1] + uexact(x[i],y[j+1],t1)/h1
						A[j-1,j-2] = -1.0/h1
					else
						A[j-1,j] = -1.0/h1
						A[j-1,j-2] = -1.0/h1
					end
				end
				A[j-1,j-1] = 2.0/dt + 2.0/h1
			end
			ut = A\b # Solve the diagonal matrix.
			for j = 1:n-1
				u1[i,j+1] = ut[j]
			end
		end # Finish y-sweep.
		t = t + dt
		
	end # Finished with the loop in time
	
	# Data analysis
	ue = similar(u1)
	for i=1:m+1
		for j=1:n+1
			ue[i,j] = uexact(x[i],y[j],tfinal)
		end
	end
	err = maximum(abs.(u1.-ue))
	
	return u1, ue, err, x, y, A, t
	
end

# ╔═╡ d8265df0-6f97-11eb-2fa1-d101834eebf8
u1_li, ue_li, err_li, x_li, y_li, A_li, t_li = li_adi(uexact, f);

# ╔═╡ d80604ae-6f97-11eb-25e9-3d48377ef4f3
err_li

# ╔═╡ c934d930-6f9b-11eb-07b0-fbf990b04f07
plt.contourf(x_li, y_li, [ue', u1_li', (u1_li.-ue_li)'], layout=(1,3), title=["Analytical" "ADI" "Error"])

# ╔═╡ Cell order:
# ╟─19de7440-6f2e-11eb-31c9-85f019d5575a
# ╠═f846cb60-6f2e-11eb-2465-4981c2a5a085
# ╠═f819eefe-6f2e-11eb-1a9e-4131d38dca4f
# ╠═f7a81a60-6f2e-11eb-39f5-9727c1bb155a
# ╠═f7918520-6f2e-11eb-3af1-03973e57e215
# ╟─f318801e-6f2e-11eb-1ee1-0165982824e6
# ╠═f3085380-6f2e-11eb-2dbc-6566d6751265
# ╠═f2f1e552-6f2e-11eb-2585-073e88af63d7
# ╠═f2dd9a00-6f2e-11eb-3e5a-f56789b381f9
# ╠═a0e23f20-6f2f-11eb-2fd7-a949d5eca41c
# ╠═9398e8f0-6f2f-11eb-0d23-054a8b5fb679
# ╠═9388e360-6f2f-11eb-34b5-351590e9a0c7
# ╠═936afb20-6f2f-11eb-2296-3d25244b1f8e
# ╟─9357e850-6f2f-11eb-2d9d-5d7f91b18c7b
# ╟─933a0012-6f2f-11eb-3035-a752ff462cc5
# ╟─891db4a0-6f34-11eb-0664-01d1220073c7
# ╟─f6f42f80-6f35-11eb-1216-49c150378a11
# ╟─d9beed60-6f99-11eb-3df0-0520d4a4ab01
# ╠═9319cde0-6f2f-11eb-118d-79b506aac9b6
# ╠═92f46b90-6f2f-11eb-0463-43567de8cef4
# ╠═f2c72bd0-6f2e-11eb-314f-1d1b3f74e1c2
# ╠═f2b09690-6f2e-11eb-0887-791d4fe4a43c
# ╠═f27e14e0-6f2e-11eb-373a-5ff0bdd189a4
# ╠═c602b840-6f3c-11eb-3749-c17f51dedfff
# ╠═f2142f80-6f2e-11eb-0d00-b169cc121d49
# ╠═f1fe84a0-6f2e-11eb-0436-2765003fd976
# ╠═eecea170-6f2e-11eb-33ad-314c0dd4bb86
# ╠═d855f970-6f97-11eb-3e40-6d89a03fb452
# ╠═d8265df0-6f97-11eb-2fa1-d101834eebf8
# ╠═d80604ae-6f97-11eb-25e9-3d48377ef4f3
# ╠═c934d930-6f9b-11eb-07b0-fbf990b04f07
