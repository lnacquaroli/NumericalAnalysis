### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 4a50f530-6bbe-11eb-174a-6dd416f7bc7f
using Compat

# ╔═╡ 6ab79822-6bbd-11eb-2ee8-3b0cab6ac249
md" ## Li - Numerical Solution to DEs

Implementation of the Successize over-relaxation (SOR($\omega$)) method to solve the Poisson equation

$$\nabla^2u(x,y)=u_{xx}+u_{yy}=f(x,y)$$

on a square domain $[a, b]\times [c, d]$ with $(b-a)=(d-c)$ and Dirichlet boundary conditions on all sides.

We test an example with true solution $u(x,y)=e^x\sin(\pi y)$, the source term then is $f(x,y)=e^x\sin(\pi y)(1-\pi^2)$. 

We take $n_x=n_y=40$, and the domain is $[-1, 1]\times [-1, 1]$.

Further function also include the solution for Neumann BC at $x=b$ keeping Dirichlet BCs on the rest.

"

# ╔═╡ 4a3779c0-6bbe-11eb-3639-67cc834e9046
@compat import LinearAlgebra as linalg

# ╔═╡ 4a1c9ec0-6bbe-11eb-0b86-b36b409b510f
@compat import SparseArrays as spars

# ╔═╡ 4a040db0-6bbe-11eb-23e3-9fcc6b6a9f38
@compat import Plots as plt

# ╔═╡ 49eeb0f0-6bbe-11eb-18b1-ad2c58c704bb
md"##### Input"

# ╔═╡ 49d5d1c0-6bbe-11eb-190c-adaaafa0949f
a, b, c, d = -1.0, 1.0, -1.0, 1.0 # Square domain

# ╔═╡ 49bddcf0-6bbe-11eb-2354-4948b8843153
nx, ny = 40, 40 # mesh points on x and y

# ╔═╡ 49a5e820-6bbe-11eb-2921-0faf3f2dc7c5
f(x,y) = exp(x)*sin(π*y)*(1.0 - π^2) # source (external) function

# ╔═╡ 498c6cb0-6bbe-11eb-3d6a-6301abbf9bcf
ue(x,y) = exp(x)*sin(π*y) # True solution

# ╔═╡ 74170d20-6bc1-11eb-11f4-0b266f3fd4d4
ω = 2.0 / (1.0 + sin(π/nx)) # optimal SOR ω

# ╔═╡ 496869f0-6bbe-11eb-0e30-230012825bd0
md"##### Call solver"

# ╔═╡ beec5650-6bbe-11eb-1cab-ef4a59e0ab39
function grid_parameters(domain, nx, ny)
	hx = (domain.x2 - domain.x1) / nx
	x  = @. domain.x1 + (0:nx) * hx
	hy = (domain.y2 - domain.y1) / ny
	y  = @. domain.y1 + (0:ny) * hy	
	return (hx=hx, hy=hy, hhx=hx^2, hhy=hy^2, hhxinv=1.0/(hx^2), hhyinv=1.0/(hy^2)), x, y
end

# ╔═╡ 28fa25e0-6bbf-11eb-034b-35d6f63c461a
function analytical_solution(ue, domain, x, y, nx, ny)
	u2 = zeros(nx+1, ny+1)
	for i in 1:nx+1, j in 1:nx+1
		u2[1,j] = ue(domain.x1, y[j])
		u2[nx+1,j] = ue(domain.x2, y[j])
		u2[i,1] = ue(x[i], domain.y1)
		u2[i,ny+1] = ue(x[i], domain.y2)
	end
	return u2
end

# ╔═╡ 30b30810-6bc9-11eb-1f99-9148a5537d56
function analytical_solution_b(ue, domain, x, y, nx, ny)
	u2 = zeros(nx+1, ny+1)
	for i in 1:nx+1, j in 1:ny+1
		u2[1,j] = ue(domain.x1, y[j])
		# u2[nx+1,i] = ue(domain.x2, y[i])
		u2[i,1] = ue(x[i], domain.y1)
		u2[i,ny+1] = ue(x[i], domain.y2)
	end
	return u2
end

# ╔═╡ 7d78e560-6bc0-11eb-0eb8-95cc5989234b
function source_function(fs, x, y, nx, ny)
	f = Array{Float64}(undef, nx+1, ny+1)
	for i in 1:nx+1, j in 1:nx+1
		f[i,j] = fs(x[i], y[j])
	end
	return f
end

# ╔═╡ eb55f752-6bbe-11eb-04ea-c54df8be6e22
function poisson_2d_sor(domain, nx, ny, fs, ue, ω; ε=1.0e-5, σ₀=Inf)
	
	h, x, y = grid_parameters(domain, nx, ny)
	
	uk = analytical_solution(ue, domain, x, y, nx, ny)
	f = source_function(fs, x, y, nx, ny)
	
	k = 0
	σ = [σ₀]
	while σ₀ > ε
		k += 1
		ukm1 = copy(uk)
		
		for i in 2:nx, j in 2:ny
			# ue2[i,j] = (1.0 - ω)*u1[i,j] - ω*((ue2[i-1,j] + ue2[i+1,j] 
			# 			+ ue2[i,j-1] + ue2[i,j+1]) - h.hhx*f[i,j])*0.25
			uk[i,j] = (1.0 - ω)*ukm1[i,j] + ω*(uk[i-1,j] + uk[i+1,j] + uk[i,j-1] + uk[i,j+1] - h.hhx*f[i,j])*0.25
		end
		σ₀ = linalg.norm(uk .- ukm1, Inf)
		push!(σ, σ₀)
	end
	
	R = 0.0 # residual
	for i in 2:nx, j in 2:ny
		R += abs((uk[i+1,j] + uk[i,j+1] + uk[i-1,j] + uk[i,j-1] - 4.0*uk[i,j])*h.hhxinv - f[i,j])
	end
	
	return x, y, uk, R, k, σ[2:end]
end

# ╔═╡ f0d38f80-6bc8-11eb-3175-b37578c55df2
function poisson_2d_sor_b(domain, nx, ny, fs, ue, ω; ε=1.0e-5, σ₀=Inf)
	
	nx == ny || throw("This function runs for n = nx = ny.")

	h, x, y = grid_parameters(domain, nx, ny)
	
	uk = analytical_solution(ue, domain, x, y, nx, ny)	
	f = source_function(fs, x, y, nx, ny)
	
	k = 0
	σ = [σ₀]
	while σ₀ > ε
		k += 1
		ukm1 = copy(uk)
		
		for i in 2:nx+1, j in 2:ny
			if i == nx+1
				uk[i,j] = (1.0 - ω)*ukm1[i,j] + ω*(2.0*uk[i-1,j] + uk[i,j-1] + uk[i,j+1] - h.hhx*f[i,j])*0.25 + 0.5*ue(domain.x2,y[j])*h.hx
			else
				uk[i,j] = (1.0 - ω)*ukm1[i,j] + ω*(uk[i-1,j] + uk[i+1,j] + uk[i,j-1] + uk[i,j+1] - h.hhx*f[i,j])*0.25
			end
		end
		σ₀ = linalg.norm(uk .- ukm1, Inf)
		push!(σ, σ₀)
	end
	
	R = 0.0 # residual
	for i in 2:nx, j in 2:ny
		R += abs((uk[i+1,j] + uk[i,j+1] + uk[i-1,j] + uk[i,j-1] - 4.0*uk[i,j])*h.hhxinv - f[i,j])
	end
	
	return x, y, uk, R, k, σ[2:end]
end

# ╔═╡ bed0b800-6bbe-11eb-2c62-016e51704bc3
domain = (x1=a, x2=b, y1=c, y2=d)

# ╔═╡ 967b0940-6c19-11eb-398e-3da1ae6a60ff
md"##### Gauss-Seidel with all Dirichlet BCs"

# ╔═╡ a4a6d030-6c19-11eb-275c-b17892e04fa9
x_gs, y_gs, u_gs, R_gs, k_gs, σ_gs = poisson_2d_sor(domain, nx, ny, f, ue, 1);

# ╔═╡ a48fc5c0-6c19-11eb-1620-61e5172ce58f
plt.plot(1:k_gs, σ_gs, title="Absolute error GS", lab="", xaxis=("Iterations", :log10), yaxis=(:log10))

# ╔═╡ a473b240-6c19-11eb-308b-99e787b9ff34
R_gs, k_gs, R_gs/nx/ny

# ╔═╡ 75309600-6c1a-11eb-08cc-61781453ff63
uexact = analytical_solution(ue, domain, x_gs, y_gs, nx, ny);

# ╔═╡ e296d4d0-6c19-11eb-24be-6b50697eecb7
plt.contourf(x_gs, y_gs, [u_gs', uexact', (u_gs.-uexact)'], layout=(1,3), title=["GS Solution" "Analytical" "Error"])

# ╔═╡ 9cfb1abe-6bca-11eb-11b4-1d3a783b4cea
md"##### SOR with all Dirichlet BCs"

# ╔═╡ beb7639e-6bbe-11eb-2520-8540fd324a15
x_sor_opt, y_sor_opt, u_sor_opt, R_sor_opt, k_sor_opt, σ_sor_opt = poisson_2d_sor(domain, nx, ny, f, ue, ω);

# ╔═╡ 49323ec0-6bbe-11eb-1c47-897258fc72e9
plt.plot(1:k_sor_opt, σ_sor_opt, title="Absolute error SOR-opt", lab="", xaxis=("Iterations", :log10), yaxis=(:log10))

# ╔═╡ 5891aa70-6c10-11eb-3a45-c57657ebbd8d
R_sor_opt, k_sor_opt, R_sor_opt/nx/ny

# ╔═╡ a825abd0-6c1b-11eb-265f-8fe9ec6c1518
plt.contourf(x_sor_opt, y_sor_opt, [u_sor_opt', uexact', (u_sor_opt.-uexact)'], layout=(1,3), title=["SOR-opt Solution" "Analytical" "Error"])

# ╔═╡ b89a7860-6c1b-11eb-3e4d-c93e1e565477
R_sor_opt, k_sor_opt, R_sor_opt/nx/ny

# ╔═╡ 483abf60-6bbe-11eb-3848-3dcd2930616a
md"##### SOR with 3 Dirichlet BCs and 1 Neumann"

# ╔═╡ 4814c0d0-6bbe-11eb-27fd-8d357a1e9aa8
x_sor_opt_b, y_sor_opt_b, u_sor_opt_b, R_sor_opt_b, k_sor_opt_b, σ_sor_opt_b  = poisson_2d_sor_b(domain, nx, ny, f, ue, ω);

# ╔═╡ 47fc7de0-6bbe-11eb-3302-252dcd80d2ad
plt.plot(1:k_sor_opt_b, σ_sor_opt_b, title="Absolute error SOR-b", lab="", xaxis=("Iterations", :log10), yaxis=(:log10))

# ╔═╡ 46ad68a0-6bbe-11eb-2c29-d5ef58c360a8
plt.contourf(x_sor_opt_b, y_sor_opt_b, [u_sor_opt_b', uexact', (u_sor_opt_b .- uexact)'], layout=(1,3), title=["SOR-opt_b Solution" "Analytical" "Error"])

# ╔═╡ 46aa3450-6bbe-11eb-1b9e-2b559ce56252
R_sor_opt_b, k_sor_opt_b, R_sor_opt_b/nx/ny

# ╔═╡ 46a6b1e0-6bbe-11eb-1663-a9de059ad84f


# ╔═╡ 46a52b40-6bbe-11eb-3085-e56c3c69436e


# ╔═╡ 46a37d92-6bbe-11eb-0f01-01e684c27156


# ╔═╡ 46a1f6ee-6bbe-11eb-1994-cbe67312d0c3


# ╔═╡ 46a02230-6bbe-11eb-2041-cbeb1162fd4f


# ╔═╡ 469e747e-6bbe-11eb-01f0-b3f698d65716


# ╔═╡ 469d3c00-6bbe-11eb-151b-a7c31c04b824


# ╔═╡ 469bb560-6bbe-11eb-3782-b7ef2a3205bd


# ╔═╡ 469a2ec0-6bbe-11eb-3361-31e2a26e884b


# ╔═╡ 4698cf30-6bbe-11eb-19e8-29adc7064f1c


# ╔═╡ 46972180-6bbe-11eb-2a77-cfdce3498abe


# ╔═╡ 46959ae0-6bbe-11eb-1329-7f0a5c961d7f


# ╔═╡ 4693ed2e-6bbe-11eb-2bc8-3fe59de77a95


# ╔═╡ 46921870-6bbe-11eb-3894-61993cb978f7


# ╔═╡ 468fce80-6bbe-11eb-06be-a591693e0c03


# ╔═╡ 468c4c10-6bbe-11eb-00ec-bfbe8d5befde


# ╔═╡ 4417c9a0-6bbe-11eb-17f2-43832e04ac85


# ╔═╡ 589a5640-6bbe-11eb-115f-3d2f60dcf545


# ╔═╡ Cell order:
# ╟─6ab79822-6bbd-11eb-2ee8-3b0cab6ac249
# ╠═4a50f530-6bbe-11eb-174a-6dd416f7bc7f
# ╠═4a3779c0-6bbe-11eb-3639-67cc834e9046
# ╠═4a1c9ec0-6bbe-11eb-0b86-b36b409b510f
# ╠═4a040db0-6bbe-11eb-23e3-9fcc6b6a9f38
# ╟─49eeb0f0-6bbe-11eb-18b1-ad2c58c704bb
# ╠═49d5d1c0-6bbe-11eb-190c-adaaafa0949f
# ╠═49bddcf0-6bbe-11eb-2354-4948b8843153
# ╠═49a5e820-6bbe-11eb-2921-0faf3f2dc7c5
# ╠═498c6cb0-6bbe-11eb-3d6a-6301abbf9bcf
# ╠═74170d20-6bc1-11eb-11f4-0b266f3fd4d4
# ╟─496869f0-6bbe-11eb-0e30-230012825bd0
# ╟─beec5650-6bbe-11eb-1cab-ef4a59e0ab39
# ╟─28fa25e0-6bbf-11eb-034b-35d6f63c461a
# ╠═30b30810-6bc9-11eb-1f99-9148a5537d56
# ╠═7d78e560-6bc0-11eb-0eb8-95cc5989234b
# ╠═eb55f752-6bbe-11eb-04ea-c54df8be6e22
# ╠═f0d38f80-6bc8-11eb-3175-b37578c55df2
# ╠═bed0b800-6bbe-11eb-2c62-016e51704bc3
# ╟─967b0940-6c19-11eb-398e-3da1ae6a60ff
# ╠═a4a6d030-6c19-11eb-275c-b17892e04fa9
# ╠═a48fc5c0-6c19-11eb-1620-61e5172ce58f
# ╠═a473b240-6c19-11eb-308b-99e787b9ff34
# ╠═75309600-6c1a-11eb-08cc-61781453ff63
# ╠═e296d4d0-6c19-11eb-24be-6b50697eecb7
# ╟─9cfb1abe-6bca-11eb-11b4-1d3a783b4cea
# ╠═beb7639e-6bbe-11eb-2520-8540fd324a15
# ╠═49323ec0-6bbe-11eb-1c47-897258fc72e9
# ╠═5891aa70-6c10-11eb-3a45-c57657ebbd8d
# ╠═a825abd0-6c1b-11eb-265f-8fe9ec6c1518
# ╠═b89a7860-6c1b-11eb-3e4d-c93e1e565477
# ╟─483abf60-6bbe-11eb-3848-3dcd2930616a
# ╠═4814c0d0-6bbe-11eb-27fd-8d357a1e9aa8
# ╠═47fc7de0-6bbe-11eb-3302-252dcd80d2ad
# ╠═46ad68a0-6bbe-11eb-2c29-d5ef58c360a8
# ╠═46aa3450-6bbe-11eb-1b9e-2b559ce56252
# ╠═46a6b1e0-6bbe-11eb-1663-a9de059ad84f
# ╠═46a52b40-6bbe-11eb-3085-e56c3c69436e
# ╠═46a37d92-6bbe-11eb-0f01-01e684c27156
# ╠═46a1f6ee-6bbe-11eb-1994-cbe67312d0c3
# ╠═46a02230-6bbe-11eb-2041-cbeb1162fd4f
# ╠═469e747e-6bbe-11eb-01f0-b3f698d65716
# ╠═469d3c00-6bbe-11eb-151b-a7c31c04b824
# ╠═469bb560-6bbe-11eb-3782-b7ef2a3205bd
# ╠═469a2ec0-6bbe-11eb-3361-31e2a26e884b
# ╠═4698cf30-6bbe-11eb-19e8-29adc7064f1c
# ╠═46972180-6bbe-11eb-2a77-cfdce3498abe
# ╠═46959ae0-6bbe-11eb-1329-7f0a5c961d7f
# ╠═4693ed2e-6bbe-11eb-2bc8-3fe59de77a95
# ╠═46921870-6bbe-11eb-3894-61993cb978f7
# ╠═468fce80-6bbe-11eb-06be-a591693e0c03
# ╠═468c4c10-6bbe-11eb-00ec-bfbe8d5befde
# ╠═4417c9a0-6bbe-11eb-17f2-43832e04ac85
# ╠═589a5640-6bbe-11eb-115f-3d2f60dcf545
