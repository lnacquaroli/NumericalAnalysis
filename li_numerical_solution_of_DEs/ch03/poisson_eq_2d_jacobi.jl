### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 4cee1d60-6c02-11eb-0c7e-c3a9556a78d7
using Compat

# ╔═╡ b9087b40-6c01-11eb-3b23-cda3c0d98571
md" ## Li - Numerical Solution to DEs

Implementation of the Jacobi method to solve the Poisson equation, 

$$\nabla^2u(x,y)=u_{xx}+u_{yy}=f(x,y)$$

on a square domain $[a, b]\times [c, d]$ with $(b-a)=(d-c)$ and Dirichlet boundary conditions on all sides.

We test an example with true solution $u(x,y)=e^x\sin(\pi y)$, the source term then is $f(x,y)=e^x\sin(\pi y)(1-\pi^2)$. 

We take $n_x=n_y=40$, and the domain is $[-1, 1]\times [-1, 1]$.

"

# ╔═╡ 4cdd5480-6c02-11eb-0c89-f3e7e04f88be
@compat import LinearAlgebra as linalg

# ╔═╡ 4cc64a10-6c02-11eb-3585-b31828d03270
@compat import SparseArrays as spars

# ╔═╡ 4cb50c00-6c02-11eb-17a6-b34b61c334ad
@compat import Plots as plt

# ╔═╡ 4c9ca200-6c02-11eb-10c5-6171b39f8232
md"##### Input"

# ╔═╡ 4c8a7990-6c02-11eb-3fa0-0966b846b2b5
a, b, c, d = -1.0, 1.0, -1.0, 1.0 # Square domain

# ╔═╡ 4c76072e-6c02-11eb-375f-a560d35c9f36
nx, ny = 40, 40 # mesh points on x and y

# ╔═╡ 4c63b7b0-6c02-11eb-2ded-373237c8c14f
f(x,y) = exp(x)*sin(π*y)*(1.0 - π^2) # source (external) function

# ╔═╡ 4c4d97a0-6c02-11eb-2086-078c269b421c
ue(x,y) = exp(x)*sin(π*y) # True solution

# ╔═╡ 4c3554b0-6c02-11eb-18fb-e54d5241f231
md"##### Call solver"


# ╔═╡ 4c250100-6c02-11eb-0032-ff309494f60f
function grid_parameters(domain, nx, ny)
	hx = (domain.x2 - domain.x1) / nx
	x  = @. domain.x1 + (0:nx) * hx
	hy = (domain.y2 - domain.y1) / ny
	y  = @. domain.y1 + (0:ny) * hy	
	return (hx=hx, hy=hy, hhx=hx^2, hhy=hy^2, hhxinv=1.0/(hx^2), hhyinv=1.0/(hy^2)), x, y
end

# ╔═╡ 4c0d8160-6c02-11eb-37e7-efd9837bdb26
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

# ╔═╡ 4bfc6a60-6c02-11eb-0bc7-c17e0607ddf5
function source_function(fs, x, y, nx, ny)
	f = Array{Float64}(undef, nx+1, ny+1)
	for i in 1:nx+1, j in 1:nx+1
		f[i,j] = fs(x[i], y[j])
	end
	return f
end

# ╔═╡ 4be4eac0-6c02-11eb-2196-1d1a8f61d5a6
function poisson_2d_jacobi(domain, nx, ny, fs, ue; ε=1.0e-5, σ₀=Inf)
	
	h, x, y = grid_parameters(domain, nx, ny)
	uk = analytical_solution(ue, domain, x, y, nx, ny)
	f = source_function(fs, x, y, nx, ny)
	
	k = 0
	σ = [σ₀]
	while σ₀ > ε
		k += 1
		ukm1 = copy(uk)	
		for i in 2:nx, j in 2:ny
			uk[i,j] = 0.25*(ukm1[i-1,j] + uk[i+1,j] + uk[i,j-1] 
						   + uk[i,j+1] - h.hhx*f[i,j]) # BC
		end
		σ₀ = linalg.norm(uk .- ukm1, Inf)
		push!(σ, σ₀)
	end
	
	R = 0.0 # residual
	for i in 2:nx, j in 2:ny
		R += abs((uk[i+1,j] + uk[i,j+1] + uk[i-1,j] + uk[i,j-1] 
				- 4.0*uk[i,j])*h.hhxinv - f[i,j])
	end
	
	return x, y, uk, R, k, σ[2:end], h
end

# ╔═╡ 4bd5cf90-6c02-11eb-363a-dbd4975bff75
domain = (x1=a, x2=b, y1=c, y2=d)

# ╔═╡ 4bbe7700-6c02-11eb-3017-372377da449a
x, y, usol, R, k, σ, h = poisson_2d_jacobi(domain, nx, ny, f, ue);

# ╔═╡ 0ce628b0-6c17-11eb-107f-4513d6b8c8fa
plt.plot(1:k, σ, title="Absolute error", lab="", xaxis=("Iterations", :log10), yaxis=(:log10))

# ╔═╡ 57bb7510-6c18-11eb-3ff8-d702499b7a90
R, k, R/nx/ny

# ╔═╡ 4bab3d1e-6c02-11eb-06f1-df2f35f09dda
uexact = analytical_solution(ue, domain, x, y, nx, ny);

# ╔═╡ 4b982a4e-6c02-11eb-25d4-81405c75f464
plt.contourf(x, y, [usol', uexact', (usol.-uexact)'], layout=(1,3), title=["Solution" "Analytical" "Error"])

# ╔═╡ 4b6c8672-6c02-11eb-2059-df2e29fc4995
plt.surface(x, y, usol', title="Solution")

# ╔═╡ 4b5a8510-6c02-11eb-28d4-a3764691c373
plt.surface(x, y, uexact', title="Analytical")

# ╔═╡ 4b43c8c0-6c02-11eb-2eae-af162c80a7ff


# ╔═╡ 4b31a050-6c02-11eb-2433-f9bcdd817b70


# ╔═╡ 4b193652-6c02-11eb-26fd-55535fd5ac3e


# ╔═╡ 4af97950-6c02-11eb-34ad-192b0decd9e9


# ╔═╡ 4ace71b0-6c02-11eb-1f78-5b7c1d461496


# ╔═╡ 4ab0b080-6c02-11eb-074f-7b460cb464a4


# ╔═╡ 4a9b2cae-6c02-11eb-3dd2-d9f35c6daf58


# ╔═╡ 4a8819de-6c02-11eb-171c-7b1de60ce351


# ╔═╡ 4a72e430-6c02-11eb-143c-f3011bfe27fd


# ╔═╡ 4a5b8ba0-6c02-11eb-260e-1d44ce94c0d7


# ╔═╡ 4a40fec0-6c02-11eb-384c-9fcd62a91404


# ╔═╡ 4a286db0-6c02-11eb-050e-111d5d009563


# ╔═╡ 2df8da30-6c02-11eb-2304-3fc0e1edf560


# ╔═╡ 2dc965c0-6c02-11eb-22c9-b726cd4e9bf4


# ╔═╡ 2d079c60-6c02-11eb-1a5c-f3452360d74f


# ╔═╡ Cell order:
# ╟─b9087b40-6c01-11eb-3b23-cda3c0d98571
# ╠═4cee1d60-6c02-11eb-0c7e-c3a9556a78d7
# ╠═4cdd5480-6c02-11eb-0c89-f3e7e04f88be
# ╠═4cc64a10-6c02-11eb-3585-b31828d03270
# ╠═4cb50c00-6c02-11eb-17a6-b34b61c334ad
# ╟─4c9ca200-6c02-11eb-10c5-6171b39f8232
# ╠═4c8a7990-6c02-11eb-3fa0-0966b846b2b5
# ╠═4c76072e-6c02-11eb-375f-a560d35c9f36
# ╠═4c63b7b0-6c02-11eb-2ded-373237c8c14f
# ╠═4c4d97a0-6c02-11eb-2086-078c269b421c
# ╟─4c3554b0-6c02-11eb-18fb-e54d5241f231
# ╟─4c250100-6c02-11eb-0032-ff309494f60f
# ╠═4c0d8160-6c02-11eb-37e7-efd9837bdb26
# ╟─4bfc6a60-6c02-11eb-0bc7-c17e0607ddf5
# ╠═4be4eac0-6c02-11eb-2196-1d1a8f61d5a6
# ╠═4bd5cf90-6c02-11eb-363a-dbd4975bff75
# ╠═4bbe7700-6c02-11eb-3017-372377da449a
# ╠═0ce628b0-6c17-11eb-107f-4513d6b8c8fa
# ╠═57bb7510-6c18-11eb-3ff8-d702499b7a90
# ╠═4bab3d1e-6c02-11eb-06f1-df2f35f09dda
# ╠═4b982a4e-6c02-11eb-25d4-81405c75f464
# ╠═4b6c8672-6c02-11eb-2059-df2e29fc4995
# ╠═4b5a8510-6c02-11eb-28d4-a3764691c373
# ╠═4b43c8c0-6c02-11eb-2eae-af162c80a7ff
# ╠═4b31a050-6c02-11eb-2433-f9bcdd817b70
# ╠═4b193652-6c02-11eb-26fd-55535fd5ac3e
# ╠═4af97950-6c02-11eb-34ad-192b0decd9e9
# ╠═4ace71b0-6c02-11eb-1f78-5b7c1d461496
# ╠═4ab0b080-6c02-11eb-074f-7b460cb464a4
# ╠═4a9b2cae-6c02-11eb-3dd2-d9f35c6daf58
# ╠═4a8819de-6c02-11eb-171c-7b1de60ce351
# ╠═4a72e430-6c02-11eb-143c-f3011bfe27fd
# ╠═4a5b8ba0-6c02-11eb-260e-1d44ce94c0d7
# ╠═4a40fec0-6c02-11eb-384c-9fcd62a91404
# ╠═4a286db0-6c02-11eb-050e-111d5d009563
# ╠═2df8da30-6c02-11eb-2304-3fc0e1edf560
# ╠═2dc965c0-6c02-11eb-22c9-b726cd4e9bf4
# ╠═2d079c60-6c02-11eb-1a5c-f3452360d74f
