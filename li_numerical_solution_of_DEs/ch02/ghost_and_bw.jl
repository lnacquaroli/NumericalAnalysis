### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 39f9c1e0-6a41-11eb-01c5-176c6fb3e43e
using Compat

# ╔═╡ 0b44aa40-6a41-11eb-2417-ab279f72c179
using Plots

# ╔═╡ d0f51a60-6a3f-11eb-3b6c-a5166391ab9c
md"## Li - Numerical Solution to DEs

Here we compare the results obtained when solving the following DE

$$u''(x) = f(x)$$

using the backward method and the ghost method (page 32).

**Input:**

$ a=0 $, $ b=1/2 $ : Two end points

$ u(x=a)=u\_a, u\_x(x=b)=u\_b $ : Dirichlet and Neumann boundary conditions

$ f(x) = -\pi^2\cos(\pi x) $: external function

$ n $: number of grid points

**Output:** 

$ x $: $ x(1), x(2), ..., x(n-1) $ are grid points 

$ U $: $ U(1), U(2),..., U(n-1) $ are approximate solutions at the grid points

"

# ╔═╡ 0b31be80-6a41-11eb-0635-9393b9f90630
@compat import LinearAlgebra as linalg

# ╔═╡ e28d3470-6a43-11eb-125a-3512ad28f2c3
@compat import SparseArrays as spars

# ╔═╡ 0b1eabb0-6a41-11eb-3d14-c7f070d4e6a3
md"##### Input"

# ╔═╡ 0b099d10-6a41-11eb-0c03-7344a15f71e2
a, b = 0.0, 0.5

# ╔═╡ 0af59fe0-6a41-11eb-00e0-05ea18e53ac6
uₐ, uₓ = 1.0, -π

# ╔═╡ 0ae12d82-6a41-11eb-3b34-5d6a02271f2b
n = [10, 20, 40, 80, 160]

# ╔═╡ 8bf4ee4e-6a43-11eb-00e8-9102e5d343aa
f(x) = -π^2 * cos(π * x) # external function

# ╔═╡ cc0e4910-6ae7-11eb-39e8-994aa773ce30
u(x) = cos(π * x) # true solution

# ╔═╡ 5576ffe0-6a42-11eb-0359-178867236760
md"##### Call solver"

# ╔═╡ d32141d0-6a42-11eb-3170-9b41049cb0a5
"""
Solves the following two-point boundary value problem: u''(x) = f(x), using center difference scheme.

Input:                                                            
    a, b: Two end points.                                            
    ua, uxb: Dirichlet and Neumann boundary conditions at a and b    
    f: external function f(x).                                       
    n: number of grid points.                                        
Output:                                                           
    x: x(1),x(2),...,x(n-1),x(n) are grid points                      
    U: U(1),U(2),...,U(n) are approximate solution at grid points 
   The method is second order accurate.
"""
function ghost_at_b(ua, uxb, f, n, h)
	hhinv = 1.0/h^2
	A = linalg.Tridiagonal(ones(n-1)*hhinv, -ones(n)*2.0*hhinv, ones(n-1)*hhinv)
	f[1] = f[1] - ua * hhinv
	f[n] = f[n] - uxb / h
	
	U = A \ f
	
	return U
end

# ╔═╡ c479d050-6a44-11eb-00ba-dd9a6330e15e
"""
Solves the following two-point boundary value problem: u''(x) = f(x), using backward  difference scheme.

Input:                                                            
    a, b: Two end points.                                            
    ua, uxb: Dirichlet and Neumann boundary conditions at a and b    
    f: external function f(x).                                       
    n: number of grid points.                                        
Output:                                                           
    x: x(1),x(2),...,x(n-1),x(n) are grid points                      
    U: U(1),U(2),...,U(n) are approximate solution at grid points 

The method is first order accurate unless uxb=0. When uxb=0, it is second order accurate.
"""
function backward_at_b(ua, uxb, f, n, h)
	hhinv = 1.0/h^2
	A = linalg.Tridiagonal(ones(n-1).*hhinv, -ones(n).*2.0*hhinv, ones(n-1).*hhinv)
	f[1] = f[1] - ua * hhinv
	f[n] = uxb / h
	
	U = A \ f
	
	return U
end

# ╔═╡ 0aa69580-6a41-11eb-3ba2-dfd0bec9a2ac
function ghost_and_bw(n, f, ua, uxb, a, b, u)
	
	h = (b - a) ./ n # grid resolution
	
	Ub = Vector{Float64}()
	Ug = Vector{Float64}()
	eb = Vector{Float64}(undef, length(n))
	eg = similar(eb)
	x = 0.0
	for k in 1:length(n)
		
		x = a .+ (1:n[k]) .* h[k] # grid points
		
		Ug = ghost_at_b(ua, uxb, f.(x), n[k], h[k])		# ghost-point method.
		Ub = backward_at_b(ua, uxb, f.(x), n[k], h[k])	# Backward Difference
		
		# Print out the maximum error
		eg[k] = linalg.norm(Ug .- u.(x), Inf)
		eb[k] = linalg.norm(Ub .- u.(x), Inf)
	end
	
	return eg, eb, Ug, Ub, h, x

end

# ╔═╡ 0a91fc10-6a41-11eb-1fe5-073eca7ef701
# Ub are the results from the backward method, Ug are the results from the ghost point method

# ╔═╡ 08271690-6a41-11eb-37ec-99b0dd0335ef
eg, eb, Ug, Ub, h, x = ghost_and_bw(n, f, uₐ, uₓ, a, b, u)

# ╔═╡ 30f16c90-6a48-11eb-3df2-0f19e6a657e0
plot(h, [eg, eb], xaxis=(:log10, (1e-4, 1)), yaxis=(:log10, (1e-6, 1)), label=["eg" "eb"], title="Grid refinement")#, aspect_ratio=:equal)

# ╔═╡ 30a59680-6a48-11eb-0a56-9bfa51fdc260
plot(x, [Ug, Ub, cos.(π.*x)], label=["Ug" "Ub" "u"], title="Solutions")

# ╔═╡ 6cd3e320-6ae6-11eb-00f0-734e6820c7c8
plot(x, [Ug.-u.(x), Ub.-u.(x)], layout=(2,1), label=["Error (Ug-u)" "Error (Ub-u)"])

# ╔═╡ 6cbadce0-6ae6-11eb-2ce8-f14c027378c4


# ╔═╡ 6c87e600-6ae6-11eb-34cb-79cf0f10dd0b


# ╔═╡ 0fab9ee0-6ae6-11eb-3882-777a27d042d8


# ╔═╡ Cell order:
# ╟─d0f51a60-6a3f-11eb-3b6c-a5166391ab9c
# ╠═39f9c1e0-6a41-11eb-01c5-176c6fb3e43e
# ╠═0b44aa40-6a41-11eb-2417-ab279f72c179
# ╠═0b31be80-6a41-11eb-0635-9393b9f90630
# ╠═e28d3470-6a43-11eb-125a-3512ad28f2c3
# ╟─0b1eabb0-6a41-11eb-3d14-c7f070d4e6a3
# ╠═0b099d10-6a41-11eb-0c03-7344a15f71e2
# ╠═0af59fe0-6a41-11eb-00e0-05ea18e53ac6
# ╠═0ae12d82-6a41-11eb-3b34-5d6a02271f2b
# ╠═8bf4ee4e-6a43-11eb-00e8-9102e5d343aa
# ╠═cc0e4910-6ae7-11eb-39e8-994aa773ce30
# ╟─5576ffe0-6a42-11eb-0359-178867236760
# ╟─d32141d0-6a42-11eb-3170-9b41049cb0a5
# ╟─c479d050-6a44-11eb-00ba-dd9a6330e15e
# ╟─0aa69580-6a41-11eb-3ba2-dfd0bec9a2ac
# ╠═0a91fc10-6a41-11eb-1fe5-073eca7ef701
# ╠═08271690-6a41-11eb-37ec-99b0dd0335ef
# ╠═30f16c90-6a48-11eb-3df2-0f19e6a657e0
# ╠═30a59680-6a48-11eb-0a56-9bfa51fdc260
# ╠═6cd3e320-6ae6-11eb-00f0-734e6820c7c8
# ╠═6cbadce0-6ae6-11eb-2ce8-f14c027378c4
# ╠═6c87e600-6ae6-11eb-34cb-79cf0f10dd0b
# ╠═0fab9ee0-6ae6-11eb-3882-777a27d042d8
