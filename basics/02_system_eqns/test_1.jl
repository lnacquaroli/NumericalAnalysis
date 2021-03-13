### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 59c8f8b0-7897-11eb-271d-0924ab514601
using Compat

# ╔═╡ 7c792100-7897-11eb-3a1f-7fb9fc5d6667
@compat import LinearAlgebra as linalg

# ╔═╡ 7c683110-7897-11eb-0bb0-6f913c43b045
function foo1!(x,i,j)
    for k in axes(x, 2)  # <- give dimension as input to axes function
        x[k, i], x[k, j] = x[k, j], x[k, i]
    end
end

# ╔═╡ 7c571a10-7897-11eb-112e-6fb2743d5ea5
A = [1 2 3;
4 5 6;
7 8 9]

# ╔═╡ 7c3f9a70-7897-11eb-3026-8f3819263012
foo1!(A, 1, 3)

# ╔═╡ 7c2a8bd0-7897-11eb-190e-9509ecc34a51
A

# ╔═╡ 7c1703d0-7897-11eb-0606-eda43d368b6c
foo1!(A,2,1)

# ╔═╡ 7c03c9f0-7897-11eb-0a46-4d902cbed7dd
A

# ╔═╡ 7bece690-7897-11eb-1629-4d29189d7e63
function call_foo1(x, i, j)
	foo1!(x,i,j)
	return x
end

# ╔═╡ 7bd95e90-7897-11eb-3182-f536729ef414
call_foo1(A, 1, 3)

# ╔═╡ 7bc699e0-7897-11eb-31b7-ff59e164bf58
A

# ╔═╡ 7bb338f0-7897-11eb-0e22-91de36294fe2
# Elimination process
function elimination!(U, f, n)
	for j in 1:n-1
		abs(U[j,j]) > eps() || throw("Zero pivot encountered.")
		for i in j+1:n
			factor = U[i,j]/U[j,j]
			for k in j+1:n # j will be zero and we are not coming back to it
				U[i,k] = U[i,k] - factor*U[j,k]
			end
			f[i] = f[i] - factor*f[j]
		end
	end
end

# ╔═╡ 7b9db522-7897-11eb-23e5-89bcaf3a3561
B = [1.0  2.0 -1.0; 2.0  1.0 -2.0; -3.0  1.0  1.0]

# ╔═╡ 7b8b8cb0-7897-11eb-20ca-f5a23de7ed77
b = [3.0; 3.0; -6.0]

# ╔═╡ 6974e510-7899-11eb-207f-f9d423d7c333
# Back-sustitution process
function back_sustitution!(x, f, U, n)
	for i in reverse(1:n)
		for j = i+1:n
			f[i] = f[i] - U[i,j]*x[j]
		end
	x[i] = f[i]/U[i,i]
	end
end

# ╔═╡ 7b02fbc0-7897-11eb-34b2-49491430852e
function naive_gaussian_elimination(A, b)
	n = size(A,1)
	U, f = copy(A), copy(b)
	x = Vector{eltype(b[1])}(undef,n)
	elimination!(U, f, n)
	back_sustitution!(x, f, U, n)
	return x, linalg.UpperTriangular(U), f
end

# ╔═╡ 7a4c7d00-7897-11eb-0a40-a71e778c7e5d
x, U, f = naive_gaussian_elimination(B, b)

# ╔═╡ 09feb500-78a0-11eb-2031-6170f8b3784a


# ╔═╡ 09caacb0-78a0-11eb-0fa9-9577706d9927


# ╔═╡ 09a104a0-78a0-11eb-2436-f195420e16d6


# ╔═╡ Cell order:
# ╠═59c8f8b0-7897-11eb-271d-0924ab514601
# ╠═7c792100-7897-11eb-3a1f-7fb9fc5d6667
# ╠═7c683110-7897-11eb-0bb0-6f913c43b045
# ╠═7c571a10-7897-11eb-112e-6fb2743d5ea5
# ╠═7c3f9a70-7897-11eb-3026-8f3819263012
# ╠═7c2a8bd0-7897-11eb-190e-9509ecc34a51
# ╠═7c1703d0-7897-11eb-0606-eda43d368b6c
# ╠═7c03c9f0-7897-11eb-0a46-4d902cbed7dd
# ╠═7bece690-7897-11eb-1629-4d29189d7e63
# ╠═7bd95e90-7897-11eb-3182-f536729ef414
# ╠═7bc699e0-7897-11eb-31b7-ff59e164bf58
# ╠═7bb338f0-7897-11eb-0e22-91de36294fe2
# ╠═7b9db522-7897-11eb-23e5-89bcaf3a3561
# ╠═7b8b8cb0-7897-11eb-20ca-f5a23de7ed77
# ╠═6974e510-7899-11eb-207f-f9d423d7c333
# ╠═7b02fbc0-7897-11eb-34b2-49491430852e
# ╠═7a4c7d00-7897-11eb-0a40-a71e778c7e5d
# ╠═09feb500-78a0-11eb-2031-6170f8b3784a
# ╠═09caacb0-78a0-11eb-0fa9-9577706d9927
# ╠═09a104a0-78a0-11eb-2436-f195420e16d6
