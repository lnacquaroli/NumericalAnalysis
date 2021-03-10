### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ f546ae92-78a1-11eb-085b-0b5931cd4ca5
using Compat

# ╔═╡ c802da30-78a1-11eb-348c-153c7b96b1bc
md"### Partial pivoting elimination method

Chapter 2, Sauer's Numerical Analysis.
"

# ╔═╡ f5326340-78a1-11eb-0e7f-a52177723af2
@compat import LinearAlgebra as linalg

# ╔═╡ 9dc87110-78a8-11eb-35f9-db921bb156d4
@compat import SparseArrays as spars

# ╔═╡ 0fd35a50-7dee-11eb-3441-d11ab871bc31
"""Forward-sustitution process"""
function _forward_substitution!(x, f, L, n)
	for i in 1:n
		x[i] = f[i]
		for j = 1:i-1
			f[i] = f[i] - L[i,j]*x[j]
		end
	end
end

# ╔═╡ 92ddf500-7de3-11eb-266c-03fb8832fbb0
"""Back-sustitution process"""
function _back_substitution!(x, f, U, n)
	for i in reverse(1:n)
		for j = i+1:n
			f[i] = f[i] - U[i,j]*x[j]
		end
		x[i] = f[i]/U[i,i]
	end
end

# ╔═╡ d34a9160-7dee-11eb-21cd-9df0909adb2f
"""Forward-sustitution process"""
function _forward_substitution(L, f, n)
	x = similar(f)
	for i in 1:n
		x[i] = f[i]
		for j = 1:i-1
			x[i] = x[i] - L[i,j]*x[j]
		end
	end
	return x
end

# ╔═╡ e826d6c0-7dee-11eb-234c-7b7b06b0c3af
"""Back-sustitution process"""
function _back_substitution(U, f, n)
	x = similar(f)
	for i in reverse(1:n)
		x[i] = f[i]
		for j = i+1:n
			x[i] = x[i] - U[i,j]*x[j]
		end
		x[i] = x[i]/U[i,i]
	end
	return x
end

# ╔═╡ f44f1d40-78a3-11eb-36fe-e7d69aa8ba97
"""PA = LU eliminiation process"""
function palu_elimination!!(U, P, n)
	for j in 1:n-1
		c = argmax(abs.(U[j:end,j])) + j - 1 # j-1 compensates de j:end
		if c > j # if a maximum was found and is bigger than j, swap rows
			linalg._swap_rows!(P, c, j)
			linalg._swap_rows!(U, c, j)
		end
		for i in j+1:n
			factor = U[i,j]/U[j,j]
			for k in j:n # j will be zero and we are not coming back to it
				U[i,k] = U[i,k] - factor*U[j,k]
			end
			U[i,j] = factor
		end	
	end
end

# ╔═╡ f466fcf0-78a1-11eb-1853-750527d71308
function palu_gaussian_elimination(A, b)
	m, n = size(A)
	m == n || throw("Coefficient matrix is not square.")
	m == length(b) || throw("The size of the coefficient matrix and rhs vector are inconsistent.")
	
	P = Matrix(1.0*linalg.I(n)) # permutation matrix
	U = copy(A) # avoid modification
	palu_elimination!!(U, P, n) # perform partial pivoting
	
	Uupp = copy(linalg.UpperTriangular(U))
	Ulow = copy(linalg.LowerTriangular(U))
	Ulow[CartesianIndex.(1:n,1:n)] .= 1.0 # unit diagonal in Lower triangular
	# for i in 1:n; Ulow[i,i] = 1.0; end
	
	# Perform palu on twice
	c = _forward_substitution(Ulow, P*b, n)
	x = _back_substitution(Uupp, c, n)
	
	return (x=x, Apa=U, f=P*b, c=c, P=P, L=Ulow, U=Uupp)
end

# ╔═╡ a92989ce-78a9-11eb-0d76-d9ae7f573f7d
A = float.([2 1 5; 4 4 -4; 1 3 1])

# ╔═╡ 41d87e60-78ab-11eb-274a-d36a7967d189
b = [5.0; 0.0; 6.0]

# ╔═╡ f1566190-78a1-11eb-2481-b5ef1f3b5f7c
sol = palu_gaussian_elimination(A, b)

# ╔═╡ 3020e6b0-78ad-11eb-269c-af7e63d03443
sol.f

# ╔═╡ c8f73160-7de8-11eb-1ade-05c01726b2c0
sol.c

# ╔═╡ 91824020-7de4-11eb-36be-03bbfcae7e10
sol.U

# ╔═╡ 9172fde0-7de4-11eb-2b6a-b954500e8407
sol.L

# ╔═╡ 915bcc60-7de4-11eb-38b1-17bf27725d81
sol.Apa

# ╔═╡ 91139fd0-7de4-11eb-02a7-53362ceb3bfe
A \ b

# ╔═╡ 9100b410-7de4-11eb-0f15-9dbaa9e8aa8f
c = _forward_substitution(sol.L, sol.P*b, 3)

# ╔═╡ 90ed5322-7de4-11eb-2dd7-dd3436e49723
_back_substitution(sol.U, c, 3)

# ╔═╡ 90db03a0-7de4-11eb-2266-632a347b7430


# ╔═╡ 90c75490-7de4-11eb-0bef-1590c95a3d1c


# ╔═╡ 90b41ab2-7de4-11eb-0b0a-b79e5a2cb2a8


# ╔═╡ 90a4ff7e-7de4-11eb-2fd8-3f6dd9d92196


# ╔═╡ 908e1c20-7de4-11eb-2187-d571991d51b3


# ╔═╡ 907def80-7de4-11eb-284d-fbc47e21e3b0


# ╔═╡ 9057c9e0-7de4-11eb-2464-8fe364a05d2e


# ╔═╡ 90433070-7de4-11eb-157c-492d9172649d


# ╔═╡ 9022fe3e-7de4-11eb-1cad-63356579a710


# ╔═╡ 901060a0-7de4-11eb-1cb6-ab3c8755979e


# ╔═╡ 8ff9562e-7de4-11eb-3d40-5dc6b04eb324


# ╔═╡ Cell order:
# ╟─c802da30-78a1-11eb-348c-153c7b96b1bc
# ╠═f546ae92-78a1-11eb-085b-0b5931cd4ca5
# ╠═f5326340-78a1-11eb-0e7f-a52177723af2
# ╠═9dc87110-78a8-11eb-35f9-db921bb156d4
# ╟─0fd35a50-7dee-11eb-3441-d11ab871bc31
# ╟─92ddf500-7de3-11eb-266c-03fb8832fbb0
# ╠═d34a9160-7dee-11eb-21cd-9df0909adb2f
# ╠═e826d6c0-7dee-11eb-234c-7b7b06b0c3af
# ╠═f44f1d40-78a3-11eb-36fe-e7d69aa8ba97
# ╠═f466fcf0-78a1-11eb-1853-750527d71308
# ╠═a92989ce-78a9-11eb-0d76-d9ae7f573f7d
# ╠═41d87e60-78ab-11eb-274a-d36a7967d189
# ╠═f1566190-78a1-11eb-2481-b5ef1f3b5f7c
# ╠═3020e6b0-78ad-11eb-269c-af7e63d03443
# ╠═c8f73160-7de8-11eb-1ade-05c01726b2c0
# ╠═91824020-7de4-11eb-36be-03bbfcae7e10
# ╠═9172fde0-7de4-11eb-2b6a-b954500e8407
# ╠═915bcc60-7de4-11eb-38b1-17bf27725d81
# ╠═91139fd0-7de4-11eb-02a7-53362ceb3bfe
# ╠═9100b410-7de4-11eb-0f15-9dbaa9e8aa8f
# ╠═90ed5322-7de4-11eb-2dd7-dd3436e49723
# ╠═90db03a0-7de4-11eb-2266-632a347b7430
# ╠═90c75490-7de4-11eb-0bef-1590c95a3d1c
# ╠═90b41ab2-7de4-11eb-0b0a-b79e5a2cb2a8
# ╠═90a4ff7e-7de4-11eb-2fd8-3f6dd9d92196
# ╠═908e1c20-7de4-11eb-2187-d571991d51b3
# ╠═907def80-7de4-11eb-284d-fbc47e21e3b0
# ╠═9057c9e0-7de4-11eb-2464-8fe364a05d2e
# ╠═90433070-7de4-11eb-157c-492d9172649d
# ╠═9022fe3e-7de4-11eb-1cad-63356579a710
# ╠═901060a0-7de4-11eb-1cb6-ab3c8755979e
# ╠═8ff9562e-7de4-11eb-3d40-5dc6b04eb324
