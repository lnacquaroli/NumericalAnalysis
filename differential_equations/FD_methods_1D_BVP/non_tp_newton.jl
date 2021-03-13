### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 39715410-69a9-11eb-3f26-e959bb431258
using Compat

# ╔═╡ 3be58860-69a9-11eb-118f-fbc2cc305c88
using Plots

# ╔═╡ b3547150-69a8-11eb-13e2-eb32bcc25d19
md" ## Li - Numerical Solution to DEs

Solve non-linear ODE 

$$u''-u^2 = f(x)$$

with $ u(0)=u(1)=0 $ and $ f(x)=-\sin(x)-\sin(x)^2 $, using Newton's method.

"

# ╔═╡ 3bd079c0-69a9-11eb-198a-f97d57a4644d
@compat import LinearAlgebra as linalg

# ╔═╡ 3bb4db70-69a9-11eb-129c-c115e377dd09
@compat import SparseArrays as spars

# ╔═╡ 3ba3eb80-69a9-11eb-38e3-7ba88bd9476e
""" Create a sparse identity matrix. """
function sparse_identity(m, n)
    return spars.sparse(1.0 * linalg.I, m, n)
end

# ╔═╡ 3f3d9790-69aa-11eb-392f-df51718a8eea
md"##### Setup parameters"

# ╔═╡ 3b8a7010-69a9-11eb-09db-59802f2c7ad5
m = 40 # mesh size

# ╔═╡ 868191c0-69a9-11eb-3f71-4f2b86fc286a
Δ = π / m # step size

# ╔═╡ 3b7365a0-69a9-11eb-2e21-fdb529b6af21
 N = m - 1

# ╔═╡ 3b53cfb0-69a9-11eb-1746-951c73159e82
x = 0.0:Δ:π # x axis

# ╔═╡ 3b20ffe0-69a9-11eb-165b-0bf150bdfa73
ue = sin.(x) # true solution

# ╔═╡ 3b090b10-69a9-11eb-09fd-872fa7a9add5
ui = x .* (π .- x)  # initial guess

# ╔═╡ 3af275d0-69a9-11eb-3966-f564d12defbc
b = @. -sin(x) - sin(x)^2 # right hand side

# ╔═╡ 3ab4a980-69a9-11eb-2c54-c17c56e69f9a
R = ones(N, 1);     # initial set-up

# ╔═╡ 3953e0fe-69a9-11eb-1121-f5112079c60d
md"##### Newton's method"

# ╔═╡ 88aa219e-69aa-11eb-1de2-4538eb3bf796
kmax = 1000

# ╔═╡ aa1aeb20-69b5-11eb-0ebc-99c1f0aaaccb
u1, F = ui[2:m], b[2:m] # remove the BCs

# ╔═╡ 88922cd0-69aa-11eb-0d39-61699a7cf18f
function newton_method(kmax, Δ, N, u1, ui, F, R; err=Inf, tol=1.0e-8)
	
	k = 0
	h = Δ^2
	hinv = 1.0 / h
	J = zeros(N, N)
	u = ui	
	
	while (err>tol) & (k<kmax)
		
		for i in 1:N
			J[i,i] = -2.0 * (1.0 + h * u1[i]) * hinv  # Jacobian diagonals
		
			# R[i] = (ui[i] - 2.0 * ui[i+1] - (h * ui[i+1])^2 + ui[i+2]) * hinv - F[i]
			R[i] = (ui[i] - 2.0 * ui[i+1] + ui[i+2]) * hinv - ui[i+1]^2 - F[i]
		end
	
		# Jacobian off diagonals
		for i in 1:N-1
			J[i,i+1] = hinv
			J[i+1,i] = hinv
		end
	
		# Jacobian alternative
		# J = ones(N,N) .* hinv
		# J[linalg.diagind(aux0, 0)] = @. -2.0 * (1.0 + h * u1) / h
		
		δ = - J \ R
		u2 = u1 .+ δ
		k += 1
		err = maximum(abs.(u2 .- u1)) # linalg.norm(u2 .- u1, Inf)
		u1 = u2
		ui[2:end-1] = u2
		
		u = [u ui]
		
	end
	
	return u, k
	
end

# ╔═╡ 6322f9a0-69ac-11eb-0f76-33c7ac25f507
u_newton, iter = newton_method(kmax, Δ, N, u1, ui, F, R; tol=1e-8)

# ╔═╡ 62dceff0-69ac-11eb-0750-d582d4febaed
plot(x, u_newton, yaxis=(0.0, 2.5), lab="")

# ╔═╡ d21c61f0-69b3-11eb-12cf-2b3fd1625002
plot(x, [u_newton[:, [1,end]], ue])

# ╔═╡ 624bac70-69ac-11eb-3bbd-3f2f554848c8
plot(x, u_newton[:,end] .- ue)

# ╔═╡ 3932eb80-69a9-11eb-1e39-81db54ec4358


# ╔═╡ 392eccd2-69a9-11eb-2fb3-556a05dc768b


# ╔═╡ 38dfc270-69a9-11eb-21a4-b7f908ae60f0


# ╔═╡ Cell order:
# ╟─b3547150-69a8-11eb-13e2-eb32bcc25d19
# ╠═39715410-69a9-11eb-3f26-e959bb431258
# ╠═3be58860-69a9-11eb-118f-fbc2cc305c88
# ╠═3bd079c0-69a9-11eb-198a-f97d57a4644d
# ╠═3bb4db70-69a9-11eb-129c-c115e377dd09
# ╠═3ba3eb80-69a9-11eb-38e3-7ba88bd9476e
# ╟─3f3d9790-69aa-11eb-392f-df51718a8eea
# ╠═3b8a7010-69a9-11eb-09db-59802f2c7ad5
# ╠═868191c0-69a9-11eb-3f71-4f2b86fc286a
# ╠═3b7365a0-69a9-11eb-2e21-fdb529b6af21
# ╠═3b53cfb0-69a9-11eb-1746-951c73159e82
# ╠═3b20ffe0-69a9-11eb-165b-0bf150bdfa73
# ╠═3b090b10-69a9-11eb-09fd-872fa7a9add5
# ╠═3af275d0-69a9-11eb-3966-f564d12defbc
# ╠═3ab4a980-69a9-11eb-2c54-c17c56e69f9a
# ╟─3953e0fe-69a9-11eb-1121-f5112079c60d
# ╠═88aa219e-69aa-11eb-1de2-4538eb3bf796
# ╠═aa1aeb20-69b5-11eb-0ebc-99c1f0aaaccb
# ╠═88922cd0-69aa-11eb-0d39-61699a7cf18f
# ╠═6322f9a0-69ac-11eb-0f76-33c7ac25f507
# ╠═62dceff0-69ac-11eb-0750-d582d4febaed
# ╠═d21c61f0-69b3-11eb-12cf-2b3fd1625002
# ╠═624bac70-69ac-11eb-3bbd-3f2f554848c8
# ╠═3932eb80-69a9-11eb-1e39-81db54ec4358
# ╠═392eccd2-69a9-11eb-2fb3-556a05dc768b
# ╠═38dfc270-69a9-11eb-21a4-b7f908ae60f0
