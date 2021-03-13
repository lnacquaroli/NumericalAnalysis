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

# ╔═╡ 9fc906b0-6d5d-11eb-25e5-eb330361aacc
using PlutoUI

# ╔═╡ bfb66680-6d52-11eb-1b20-990792939c7b
using Compat

# ╔═╡ b195ea30-6d52-11eb-2c98-55da0a9d7af2
md" ## Li - Numerical Solution to DEs

Implementation of the Crank-Nicolson method to solve the 1d heat equation, 

$$u_t = \beta u_{xx} + f(x,t),\quad a<x<b$$

$$\text{BC1:}\, u(a,t)=g_1(t),\quad \text{BC2:}\, u(b,t)=g_2(t),\quad \text{IC:}\, u(x,0)=u_0(x).$$

We take the initial estimation as the true solution $u(x,t)=\sin(\pi x)\cos(t)$, i.e., $u_0(x)=\sin(\pi x)$, and the source term $f(x,t)=-\sin(\pi x) \sin(t) + \pi^2 \sin(\pi x) \cos(t)$.

"

# ╔═╡ bfa54f80-6d52-11eb-1313-5dbc4be3590d
@compat import Plots as plt

# ╔═╡ bf910430-6d52-11eb-1cb6-459afb744ac6
@compat import LinearAlgebra as linalg

# ╔═╡ bf7c91d0-6d52-11eb-35f5-7707bed66e27
@compat import SparseArrays as spars

# ╔═╡ bf67f862-6d52-11eb-28c9-0576754fddc1
md"##### Input"

# ╔═╡ bf518a30-6d52-11eb-0e5f-bbfdeba33720
a, b = 0.0, 1.0  # Region domain

# ╔═╡ bf479f20-6d52-11eb-335d-3b72a329914a
nx = 40 # grid size

# ╔═╡ bf31a620-6d52-11eb-1db5-49f78cd3c04b
uexact(x,t) = cos(t)*cos(π*x) # true solution

# ╔═╡ be37dcd0-6d52-11eb-2417-8fa9aded341b
f(x,t) = cos(π*x)*(cos(t)*π^2 - sin(t)) # source function

# ╔═╡ be3592e0-6d52-11eb-2fea-a9091bde6484
g1(t) = cos(t) # BC at x = a

# ╔═╡ be33be1e-6d52-11eb-0afa-b16d4b22ef87
g2(t) = - cos(t) # BC at x = b

# ╔═╡ be321070-6d52-11eb-1b3d-7123020f1719
t_final = 5.5 # final time

# ╔═╡ be3062c0-6d52-11eb-1a33-958c03909750
β = 1 # diffusivity

# ╔═╡ f51661d2-6d53-11eb-21c0-d7ef8c414cb5


# ╔═╡ be2f2a40-6d52-11eb-208a-47cdefac9d61
md"##### Calculations"

# ╔═╡ be2e18d0-6d52-11eb-3b94-5deff7bff43d
function map_function(f, x, t)
	fs = [f(x[i],t[j]) for i in 1:length(x), j in 1:length(t)]		
	return fs
end

# ╔═╡ 71fe21a0-6d55-11eb-3fb5-47ba813a0b65
function matrix_coefficients(nx, h, Δt)
	dm1 = ones(nx).*(-0.5/h^2)
	d0 = ones(nx+1).*(1.0/Δt + 1.0/h^2)
	dp1 = ones(nx).*(-0.5/h^2)
	A = linalg.Tridiagonal(dm1, d0, dp1)
	A[1,:] .= 0.0 # zero first row
	A[1,1] = 1.0
	A[end,:] .= 0.0 # zero last row
	A[end, end] = 1.0
	return A
end

# ╔═╡ e28544f0-6d53-11eb-0835-27c5903a3498
h = (b - a)/nx

# ╔═╡ 1a4fa600-6d54-11eb-13e8-6f147dc049d6
Δt = h

# ╔═╡ be0979d0-6d52-11eb-0a68-13fa033e52b1
nt = Int(round(t_final/Δt))

# ╔═╡ be07cc20-6d52-11eb-167f-d9fda9b3b074
x = @. a + (0:nx)*h

# ╔═╡ c6f397c0-6d56-11eb-04ba-0114f002f1ff
t = (0:nt).*Δt

# ╔═╡ be058230-6d52-11eb-2b64-49956c739899
u0 = map_function(uexact, x, 0.0); # initial condition

# ╔═╡ be024de0-6d52-11eb-298b-fdea94220f3b
A = matrix_coefficients(nx, h, Δt) # matrix coefficients with Dirichlet

# ╔═╡ bdb69ee0-6d52-11eb-3a53-1d48826af2a4
function crank_nicolson(A, x, t, nx, nt, f, g1, g2, u0)
	hhinv = 1.0/h^2
	hhinvh = hhinv*0.5
	Δtinv = 1.0/Δt
	
	fs = map_function(f, x, t)
	b = zeros(nx+1)
	
	u = zeros(nx+1,nt+1)
	u[:, 1] = u0 # initial condition
	for j in 2:nt+1
		for i in 2:nx
			b[i] = u[i,j-1]*Δtinv + hhinvh*(u[i-1,j-1] - 2.0*u[i,j-1] + u[i+1,j-1]) + 0.5*(fs[i,j] + fs[i,j-1])
		end
		
		b[1] = g1(t[j]) 	# BC at x=a
		b[end] = g2(t[j])   # BC at x=b
	
		u[:,j] = A \ b
	end
	
	return u
end

# ╔═╡ 8408f780-6d55-11eb-3d4c-a135fd65fc20
ue = map_function(uexact, x, t);

# ╔═╡ 22d76660-6d5d-11eb-342a-2bc51acb11cc
u = crank_nicolson(A, x, t, nx, nt, f, g1, g2, u0);

# ╔═╡ 83f680f0-6d55-11eb-3df8-71e7eb6c4dc1
err = linalg.norm(u.-ue, Inf)

# ╔═╡ 83ce5f80-6d55-11eb-3896-6385ab4a653d
plt.contourf(x, t, [ue', u', (ue.-u)'], layout=(1,3), title=["Analytical" "CN" "Error"], xaxis=((0.0, 1.0)), yaxis=((0.0, 6.0)))

# ╔═╡ 83bfe090-6d55-11eb-070b-f79e15fd1771
@bind i PlutoUI.Clock()#Slider(1:length(t))

# ╔═╡ 30a036b0-6d66-11eb-3790-415645d8580b
i, t[i]

# ╔═╡ 83ad9110-6d55-11eb-1472-d31cf731046f
plt.plot(x, [ue[:,i], u[:,i]], line=([:path :scatter]), label=["Analytical" "C-N"])

# ╔═╡ 8392dd20-6d55-11eb-1749-63d198288793


# ╔═╡ 834902e0-6d55-11eb-3ea2-ad507dc4238b
function li_crank()
	a, b, m = 0.0, 1.0, 40
	h = (b-a)/m
	k = h
	h1 = h*h
	tfinal = 5.5
	n=Int(round(tfinal/k))
	
	uexact(t,x) = cos(t)*cos(pi*x)
	f(t,x) = cos(π*x)*(cos(t)*π^2 - sin(t))
	g1(t) = cos(t)
	g2(t) = -cos(t)
	
	t = 0
	x = zeros(m+1)
	u0 = similar(x)
	for i=1:m+1
		x[i] = a + (i-1)*h
		u0[i] = uexact(t,x[i])
	end
	
	A = spars.spzeros(m+1,m+1)
	for i=2:m
		A[i,i] = 1/k+1/h1
		A[i,i-1] = -0.5/h1
		A[i,i+1] = -0.5/h1
	end
	A[1,1] = 1.0
	A[m+1,m+1] = 1.0
	b = zeros(m+1)
	
	u1 = similar(b)
	for j=1:n
		for i=2:m
			b[i] = u0[i]/k + 0.5*(u0[i-1]-2.0*u0[i]+u0[i+1])/h1 + f(t+0.5*k,x[i])
		end
		b[1] = g1(t+k)   #                      % Dirichlet BC at x =a.
		b[end] = g2(t+k) #                     % Dirichlet BC at x =b.
		
		u1 = A\b
		t = t + k
		u0 = u1
	end
	
	return x, t, u1, A
	
end

# ╔═╡ 098c5b90-6d65-11eb-00cf-0df0e84af158


# ╔═╡ Cell order:
# ╟─b195ea30-6d52-11eb-2c98-55da0a9d7af2
# ╠═9fc906b0-6d5d-11eb-25e5-eb330361aacc
# ╠═bfb66680-6d52-11eb-1b20-990792939c7b
# ╠═bfa54f80-6d52-11eb-1313-5dbc4be3590d
# ╠═bf910430-6d52-11eb-1cb6-459afb744ac6
# ╠═bf7c91d0-6d52-11eb-35f5-7707bed66e27
# ╟─bf67f862-6d52-11eb-28c9-0576754fddc1
# ╠═bf518a30-6d52-11eb-0e5f-bbfdeba33720
# ╠═bf479f20-6d52-11eb-335d-3b72a329914a
# ╠═bf31a620-6d52-11eb-1db5-49f78cd3c04b
# ╠═be37dcd0-6d52-11eb-2417-8fa9aded341b
# ╠═be3592e0-6d52-11eb-2fea-a9091bde6484
# ╠═be33be1e-6d52-11eb-0afa-b16d4b22ef87
# ╠═be321070-6d52-11eb-1b3d-7123020f1719
# ╠═be3062c0-6d52-11eb-1a33-958c03909750
# ╠═f51661d2-6d53-11eb-21c0-d7ef8c414cb5
# ╟─be2f2a40-6d52-11eb-208a-47cdefac9d61
# ╠═be2e18d0-6d52-11eb-3b94-5deff7bff43d
# ╠═71fe21a0-6d55-11eb-3fb5-47ba813a0b65
# ╠═e28544f0-6d53-11eb-0835-27c5903a3498
# ╠═1a4fa600-6d54-11eb-13e8-6f147dc049d6
# ╠═be0979d0-6d52-11eb-0a68-13fa033e52b1
# ╠═be07cc20-6d52-11eb-167f-d9fda9b3b074
# ╠═c6f397c0-6d56-11eb-04ba-0114f002f1ff
# ╠═be058230-6d52-11eb-2b64-49956c739899
# ╠═be024de0-6d52-11eb-298b-fdea94220f3b
# ╠═bdb69ee0-6d52-11eb-3a53-1d48826af2a4
# ╠═8408f780-6d55-11eb-3d4c-a135fd65fc20
# ╠═22d76660-6d5d-11eb-342a-2bc51acb11cc
# ╠═83f680f0-6d55-11eb-3df8-71e7eb6c4dc1
# ╠═83ce5f80-6d55-11eb-3896-6385ab4a653d
# ╠═83bfe090-6d55-11eb-070b-f79e15fd1771
# ╠═30a036b0-6d66-11eb-3790-415645d8580b
# ╠═83ad9110-6d55-11eb-1472-d31cf731046f
# ╠═8392dd20-6d55-11eb-1749-63d198288793
# ╠═834902e0-6d55-11eb-3ea2-ad507dc4238b
# ╠═098c5b90-6d65-11eb-00cf-0df0e84af158
