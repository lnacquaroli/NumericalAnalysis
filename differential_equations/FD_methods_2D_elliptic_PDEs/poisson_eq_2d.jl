### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 02b9a190-6b14-11eb-17a6-2f27655da36a
using Compat

# ╔═╡ 3e34a3fe-6b13-11eb-1e1d-431fe21674db
md" ## Li - Numerical Solution to DEs

Here we solve the Poisson equation, $\nabla^2u(x,y)=u_{xx}+u_{yy}=f(x,y)$, on a rectangular domain $[a, b]\times [c, d]$ with a Neumann boundary condition on $x=a$ and Dirichelt boundary conditions on the other three sides. 

The mesh parameters are $m$ and $n$ and the total number of unknowns is $M=(n-1)m$.

The conversion between the 1D solution $U(k)$ and 2D array using the natural row ordering is $k=i+(j-1)m$.

"

# ╔═╡ 029b6b32-6b14-11eb-1efb-7b5f34a6f7df
@compat import LinearAlgebra as linalg

# ╔═╡ 0272fba0-6b14-11eb-3e89-b524630b13a2
@compat import SparseArrays as spars

# ╔═╡ 02553a70-6b14-11eb-116b-45db1d09c1c9
@compat import Plots as plt

# ╔═╡ 0237793e-6b14-11eb-0e54-8bfe05d8b1fb
md"##### Input"

# ╔═╡ 021ff9a0-6b14-11eb-3aaa-dbc96e26a809
a, b, c, d = 1.0, 2.0, -1.0, 1.0 # Rectangular domain

# ╔═╡ 0206f360-6b14-11eb-24ec-19aa820dc5a7
nx, ny = 32, 64 # mesh points on x and y

# ╔═╡ 2477ac40-6b15-11eb-1840-c998f2b1afed
uxb(y) = - sin(π*y)/exp(1.0) # Neumann BC at x = b

# ╔═╡ 01ee3b42-6b14-11eb-39fa-b1f9cca69f4e
f(x,y) = exp(-x)*sin(π*y)*(1.0 - π^2) # source (external) function 

# ╔═╡ 01d4e6e0-6b14-11eb-3682-e9a99b3f235c
ue(x,y) = exp(-x)*sin(π*y) # True solution

# ╔═╡ 01b7e900-6b14-11eb-379d-370b270527e6
md"##### Call solver"

# ╔═╡ 018d568e-6b14-11eb-3b2a-51ad5c218342
function grid_parameters(domain, nx, ny)
	hx = (domain.x2 - domain.x1) / nx
	x  = @. domain.x1 + (0:nx) * hx
	hy = (domain.y2 - domain.y1) / ny
	y  = @. domain.y1 + (0:ny) * hy	
	return (hx=hx, hy=hy, hhx=hx^2, hhy=hy^2, hhxinv=1.0/(hx^2), hhyinv=1.0/(hy^2)), x, y
end

# ╔═╡ 4333ec00-6b3f-11eb-2d8b-0d79d31b5754
function poisson_2d(domain, nx, ny, f, ux, ue)
	
	h, x, y = grid_parameters(domain, nx, ny)

	M = (ny - 1)*nx
	A = spars.spzeros(M, M)
	bf = zeros(M)	
	
	for j in 1:ny-1, i in 1:nx
		
		k = i + (j - 1)*nx
		
		bf[k] = f(x[i], y[j+1])
		A[k,k] = -2.0*h.hhxinv - 2.0*h.hhyinv
		
		# x-axis
		if i == 1
			A[k,k+1] = 2.0*h.hhxinv
			bf[k] = bf[k] + 2.0*ux(y[j+1])/h.hx # BC-N, check h.hx is not h.hhxinv
		else
			if i == nx
				A[k,k-1] = h.hhxinv
				bf[k] = bf[k] - ue(x[i+1],y[j+1])*h.hhxinv # BC-D at x=b
			else
				A[k,k-1] = h.hhxinv
				A[k,k+1] = h.hhxinv
			end
		end
		
		# y-axis
		if j == 1
			A[k,k+nx] = h.hhyinv
			bf[k] = bf[k] - ue(x[i], domain.y1)*h.hhyinv # BC-D at y=y1=c
		else
			if j == ny-1
				A[k,k-nx] = h.hhyinv
				bf[k] = bf[k] - ue(x[i],domain.y2)*h.hhyinv # BC-D at y=y2=d
			else
				A[k,k-nx] = h.hhyinv
				A[k,k+nx] = h.hhyinv
			end
		end
	end
	
	U1d = A \ bf # Solve system
	
	# Transform U in 1D back to (i,j) 2D form to plot the solution
	U2d = reshape(U1d, nx, ny-1)
	ue2d = [ue(x[i], y[j]) for i in 1:nx, j in 1:ny]
	
	err = linalg.norm(U2d .- ue2d[:, 2:ny], Inf) # maximum error
	
	return x[1:nx], y[2:ny], U2d, err, A, bf, U1d, h, ue2d[:, 2:ny]
end

# ╔═╡ a060eb30-6b3f-11eb-1fb0-ff502632085d
domain = (x1=a, x2=b, y1=c, y2=d)

# ╔═╡ 42a8c300-6b3f-11eb-1117-79d4ee4f29e8
x, y, U2d, err, A, bf, U1d, h, ue2d = poisson_2d(domain, nx, ny, f, uxb, ue)

# ╔═╡ 41a0eff2-6b3f-11eb-027b-0d7bdaa5e61a
plt.surface(x, y, U2d', title="Solution")

# ╔═╡ 413ab410-6b3f-11eb-271c-71b898469760
plt.plot(x, y, (U2d .- ue2d)', title="Error", seriestype=:surface)

# ╔═╡ 408ff520-6b3f-11eb-13cf-919275aa3068


# ╔═╡ Cell order:
# ╟─3e34a3fe-6b13-11eb-1e1d-431fe21674db
# ╠═02b9a190-6b14-11eb-17a6-2f27655da36a
# ╠═029b6b32-6b14-11eb-1efb-7b5f34a6f7df
# ╠═0272fba0-6b14-11eb-3e89-b524630b13a2
# ╠═02553a70-6b14-11eb-116b-45db1d09c1c9
# ╟─0237793e-6b14-11eb-0e54-8bfe05d8b1fb
# ╠═021ff9a0-6b14-11eb-3aaa-dbc96e26a809
# ╠═0206f360-6b14-11eb-24ec-19aa820dc5a7
# ╠═2477ac40-6b15-11eb-1840-c998f2b1afed
# ╠═01ee3b42-6b14-11eb-39fa-b1f9cca69f4e
# ╠═01d4e6e0-6b14-11eb-3682-e9a99b3f235c
# ╟─01b7e900-6b14-11eb-379d-370b270527e6
# ╠═018d568e-6b14-11eb-3b2a-51ad5c218342
# ╠═4333ec00-6b3f-11eb-2d8b-0d79d31b5754
# ╠═a060eb30-6b3f-11eb-1fb0-ff502632085d
# ╠═42a8c300-6b3f-11eb-1117-79d4ee4f29e8
# ╠═41a0eff2-6b3f-11eb-027b-0d7bdaa5e61a
# ╠═413ab410-6b3f-11eb-271c-71b898469760
# ╠═408ff520-6b3f-11eb-13cf-919275aa3068
