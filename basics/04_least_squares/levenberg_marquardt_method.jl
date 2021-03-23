### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 045d5310-873a-11eb-1fba-37de6c85f4d7
using Compat

# ╔═╡ f770c2e0-8739-11eb-0678-5b3d63a2ddaa
md"### Levenberg–Marquardt Method

Chapter 4 - Sauer's Numerical Analysis"

# ╔═╡ 0445fa80-873a-11eb-1b6b-9d3184d2e891
@compat import Plots as plt

# ╔═╡ 6c6e25a0-87f4-11eb-04ab-d7689e3c5d8e
@compat import LsqFit as ls

# ╔═╡ a6de5160-873b-11eb-2981-4518ed8195f6
@compat import LinearAlgebra as linalg

# ╔═╡ 22520650-88d3-11eb-2aa6-3d04a6b41b8c
function check_bounds!(p1, ub, lb, m)
	for j in 1:m
		if p1[j] < lb[j]
			p1[j] = lb[j]
		elseif p1[j] > ub[j] 
			p1[j] = ub[j]
		end
	end
end

# ╔═╡ 04313a00-873a-11eb-2278-ad216335b516
function levenberg_marquardt(r, Dr, λ, p; λm=1.0, tol=1.0e-6, k=100, ub=10.0.*ones(length(p)), lb=zeros(length(p)))
	p0 = float.(copy(vec(p)))
	p1 = copy(p0)
	ε = [Inf]
	A = []
	i = 1
	m = length(p0)
	while i < k
		i += 1
		A = Dr(p0)
		AA = A'*A
		v = (AA .+ λ.*linalg.Diagonal(AA)) \ (-A'*r(p0))
		p1 .+= v
		check_bounds!(p1, ub, lb, m)
		push!(ε, sum(r(p1).^2)) # r(p1)'*r(p1)
		(abs(ε[i] - ε[i-1]) ≤ tol) && return (x=p0, iter=i, J=A, res=r(p0), err=float.(ε[2:end]))
		# Update λ parameter
		if ε[i] < ε[i-1] 
			λ /= λm
			p0 = copy(p1)	
		else
			λ *= λm
			splice!(ε, i)
			i -= 1
		end
	end
	return (x=p0, iter=k, J=A, res=r(p0), err=float.(ε[2:end]))
end

# ╔═╡ e16d9440-873a-11eb-0b07-3d77c9c06389
t = [1,2,2,3,4]

# ╔═╡ f5091c42-873a-11eb-3cf2-65b3e7412d93
y = [3,5,7,5,1]

# ╔═╡ 038a72b2-873a-11eb-2bc7-579a7e44086e
@. f(x,p) = p[1]*exp(-p[2]*(x-p[3])^2)

# ╔═╡ 9b6e20a0-873e-11eb-1824-4beeedca14c1
p0 = [1,1,1]

# ╔═╡ 03775fe2-873a-11eb-0be2-3b4f9c0285f4
r(p) = [f(t[j],p)-y[j] for j ∈ 1:5]

# ╔═╡ ad435790-87ee-11eb-2bbc-6751a71155dd
r(p0)

# ╔═╡ 2e002620-87ee-11eb-3a63-ddcb4b670770
Dr1(p) = exp.(-p[2]*(t.-p[3]).^2)

# ╔═╡ 39f2029e-87ee-11eb-3615-05e081530b5e
Dr1(p0)

# ╔═╡ 5db6f650-87ee-11eb-09ec-cbdd3f586f98
Dr2(p) = -p[1]*(t.-p[3]).^2 .* exp.(-p[2]*(t.-p[3]).^2)

# ╔═╡ 6d556010-87ee-11eb-0a5e-5f74f8933325
Dr2(p0)

# ╔═╡ 79055190-87ee-11eb-36de-4734c96c02d3
Dr3(p) = 2.0*p[1]*p[2].*(t.-p[3]).*exp.(-p[2].*(t.-p[3]).^2)

# ╔═╡ 2c718c50-87ed-11eb-3431-316c4bf5b14d
Dr3(p0)

# ╔═╡ 917bd4b0-87ee-11eb-00f2-a557fb425638
Dr(p) = [Dr1(p) Dr2(p) Dr3(p)]

# ╔═╡ ef778070-87ec-11eb-2672-01765b1dd566
Dr(p0)

# ╔═╡ 37243210-87ed-11eb-0af0-09c509affbdb
sol = levenberg_marquardt(r, Dr, 50, [1.0,1.0,1.0]; k=1000)#, lb=[5.0,0.0,1.5], ub=[7.0, 1.0, 5.0])

# ╔═╡ 34a56c00-88d5-11eb-3fb1-d3f88a1d14c8
plt.plot(sol.err)

# ╔═╡ 3da92940-88d5-11eb-3737-0fe3271e2eb2
sol.err[end]-sol.err[end-1]

# ╔═╡ 8d4bee20-88d4-11eb-038d-b30fb51d4508
plt.plot(0:0.1:5, f(0:0.1:5, sol.x))
plt.scatter!(t,y)

# ╔═╡ 8df0e0b0-88d4-11eb-0e84-8762a32ba4e1
md"Try with `LsqFit.jl` module"

# ╔═╡ 503f94d0-87fa-11eb-1564-dd3f640a44a3
@. model(x, p) = p[1]*exp(-p[2]*(x-p[3])^2)

# ╔═╡ 502c0cd0-87fa-11eb-057d-7b8eb39b345e
fit = ls.curve_fit(model, float.(t), float.(y), float.(p0))

# ╔═╡ 4ff9d940-87fa-11eb-13af-938cfe10863f
fit.param

# ╔═╡ 4fe0d300-87fa-11eb-32e8-a367d8f1e0a4
plt.plot(0:0.1:5, model(0:0.1:5, fit.param))
plt.scatter!(t,y)

# ╔═╡ 4fc69442-87fa-11eb-39b9-6f2d804b6d6f
r(fit.param)

# ╔═╡ 4f9746e0-87fa-11eb-278d-ef7903cb4851


# ╔═╡ Cell order:
# ╟─f770c2e0-8739-11eb-0678-5b3d63a2ddaa
# ╠═045d5310-873a-11eb-1fba-37de6c85f4d7
# ╠═0445fa80-873a-11eb-1b6b-9d3184d2e891
# ╠═6c6e25a0-87f4-11eb-04ab-d7689e3c5d8e
# ╠═a6de5160-873b-11eb-2981-4518ed8195f6
# ╠═22520650-88d3-11eb-2aa6-3d04a6b41b8c
# ╠═04313a00-873a-11eb-2278-ad216335b516
# ╠═e16d9440-873a-11eb-0b07-3d77c9c06389
# ╠═f5091c42-873a-11eb-3cf2-65b3e7412d93
# ╠═038a72b2-873a-11eb-2bc7-579a7e44086e
# ╠═9b6e20a0-873e-11eb-1824-4beeedca14c1
# ╠═03775fe2-873a-11eb-0be2-3b4f9c0285f4
# ╠═ad435790-87ee-11eb-2bbc-6751a71155dd
# ╠═2e002620-87ee-11eb-3a63-ddcb4b670770
# ╠═39f2029e-87ee-11eb-3615-05e081530b5e
# ╠═5db6f650-87ee-11eb-09ec-cbdd3f586f98
# ╠═6d556010-87ee-11eb-0a5e-5f74f8933325
# ╠═79055190-87ee-11eb-36de-4734c96c02d3
# ╠═2c718c50-87ed-11eb-3431-316c4bf5b14d
# ╠═917bd4b0-87ee-11eb-00f2-a557fb425638
# ╠═ef778070-87ec-11eb-2672-01765b1dd566
# ╠═37243210-87ed-11eb-0af0-09c509affbdb
# ╠═34a56c00-88d5-11eb-3fb1-d3f88a1d14c8
# ╠═3da92940-88d5-11eb-3737-0fe3271e2eb2
# ╠═8d4bee20-88d4-11eb-038d-b30fb51d4508
# ╟─8df0e0b0-88d4-11eb-0e84-8762a32ba4e1
# ╠═503f94d0-87fa-11eb-1564-dd3f640a44a3
# ╠═502c0cd0-87fa-11eb-057d-7b8eb39b345e
# ╠═4ff9d940-87fa-11eb-13af-938cfe10863f
# ╠═4fe0d300-87fa-11eb-32e8-a367d8f1e0a4
# ╠═4fc69442-87fa-11eb-39b9-6f2d804b6d6f
# ╠═4f9746e0-87fa-11eb-278d-ef7903cb4851
