### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 5d8b0ef0-7d67-11eb-00f9-2933fd48671f
using Compat

# ╔═╡ becc6ca2-7d61-11eb-0d67-5fc765579ade
md"### Multivariate Newton’s Method

Chapter 2, Sauer's Numerical Analysis.
"

# ╔═╡ 610d7ef0-7d67-11eb-25c6-f1888bca50a1
@compat import LinearAlgebra as linalg

# ╔═╡ e45f8fa0-7d6c-11eb-32e1-354b9128b37c
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

# ╔═╡ e3353f00-7df1-11eb-3b54-d7f0837ee2b1
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

# ╔═╡ 4d4b4a30-7d69-11eb-152e-8df66b689139
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

# ╔═╡ 8a5b5340-7d62-11eb-3263-97a1a7211142
function _palu_gaussian_elimination(A, b, n)
	
	Apa = copy(A)
	P = Matrix(1.0*linalg.I(n)) # permutation matrix
	
	palu_elimination!!(Apa, P, n)
	
	U = copy(linalg.UpperTriangular(Apa))
	L = copy(linalg.LowerTriangular(Apa))
	L[CartesianIndex.(1:n,1:n)] .= 1.0
	# for i in 1:n; L[i,i] = 1.0; end
	
	c = _forward_substitution(L, P*b, n)
	x = _back_substitution(U, c, n)
	
	return x
end

# ╔═╡ dfc5e440-7d61-11eb-0a68-85015784f2f3
function multivariate_newton_method(F, J, x₀; ε=1.0e-8, k=10)
	x0 = float.(copy(vec(x₀)))
	m, n = size(J(x0...))
	x1 = copy(x0)
	# m == n || throw("Coefficient matrix is not square.")
	# m == length(b) || throw("The size of the coefficient matrix and rhs vector are inconsistent.")
	for i = 1:k
		s = _palu_gaussian_elimination(J(x0...), -F(x0...), n)
		x1 .+= s
		@show s
		@show x1
		linalg.norm(x1.-x0, Inf) > ε || return (x=x0, iter=i, err=linalg.norm(x1.-x0, Inf))
		x0 = x1
	end
	return (x=x0, iter=k, err=linalg.norm(x1.-x0, Inf))
end

# ╔═╡ dfb171e0-7d61-11eb-1301-c3853feadf32
F(u,v) = [v - u^3; u^2 + v^2 - 1]

# ╔═╡ df9f7080-7d61-11eb-35be-33e2769fe7a5
J(u,v) = [-3*u^2 1; 2*u 2*v]

# ╔═╡ df8b2530-7d61-11eb-3b21-b3588a6de988
x0 = [1, 2]

# ╔═╡ df2f2280-7d61-11eb-157a-23cbd4c8182f
sol = multivariate_newton_method(F, J, x0; k=5)

# ╔═╡ 1677d3a0-7d66-11eb-11e1-c1da8a7ef851
sol.x

# ╔═╡ 162fce20-7d66-11eb-29b6-ffc095dbec6c


# ╔═╡ 1618eac0-7d66-11eb-0177-472553836f2c


# ╔═╡ 15fe5dde-7d66-11eb-1214-076a789d3017


# ╔═╡ 15e3f810-7d66-11eb-0371-b3e46f87e733


# ╔═╡ Cell order:
# ╟─becc6ca2-7d61-11eb-0d67-5fc765579ade
# ╠═5d8b0ef0-7d67-11eb-00f9-2933fd48671f
# ╠═610d7ef0-7d67-11eb-25c6-f1888bca50a1
# ╠═e45f8fa0-7d6c-11eb-32e1-354b9128b37c
# ╠═e3353f00-7df1-11eb-3b54-d7f0837ee2b1
# ╠═4d4b4a30-7d69-11eb-152e-8df66b689139
# ╠═8a5b5340-7d62-11eb-3263-97a1a7211142
# ╠═dfc5e440-7d61-11eb-0a68-85015784f2f3
# ╠═dfb171e0-7d61-11eb-1301-c3853feadf32
# ╠═df9f7080-7d61-11eb-35be-33e2769fe7a5
# ╠═df8b2530-7d61-11eb-3b21-b3588a6de988
# ╠═df2f2280-7d61-11eb-157a-23cbd4c8182f
# ╠═1677d3a0-7d66-11eb-11e1-c1da8a7ef851
# ╠═162fce20-7d66-11eb-29b6-ffc095dbec6c
# ╠═1618eac0-7d66-11eb-0177-472553836f2c
# ╠═15fe5dde-7d66-11eb-1214-076a789d3017
# ╠═15e3f810-7d66-11eb-0371-b3e46f87e733
