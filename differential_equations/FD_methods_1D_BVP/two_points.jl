### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ cfc2b940-67ec-11eb-07e6-2d8f575d0e14
using Compat

# ╔═╡ cfabaed0-67ec-11eb-2dfd-c3c6cdf2b692
using Plots

# ╔═╡ f5ac2580-67ea-11eb-1e9c-272c12b93242
md" ## Li - Numerical solution to DEs

The function `two_points` solves the following two-points boundary value problem:

$$u''(x) = f(x)$$

using the centered finite difference scheme

Input:

$ a $, $ b $ : Two end points

$ u(x=a)=u\_a, u(x=b)=u\_b $ : Dirichlet boundary conditions

$ f $: external function $ f(x) $

$ n $: number of grid points

Output: 

$ x $: $ x(1), x(2), ..., x(n-1) $ are grid points 

$ U $: $ U(1), U(2),..., U(n-1) $ are approximate solution at grid points

"

# ╔═╡ cf94f280-67ec-11eb-14e0-dfefde71afaa
@compat import LinearAlgebra as linalg

# ╔═╡ cf7a8cb0-67ec-11eb-2e7b-83f9241fbe11
@compat import SparseArrays as spars

# ╔═╡ cf672bc0-67ec-11eb-2ab0-2bd22cafff34
""" Create a sparse identity matrix. """
function sparse_identity(m, n)
    return spars.sparse(1.0 * linalg.I, m, n)
end

# ╔═╡ cf5180e0-67ec-11eb-1cfd-1d3026c131c8
""" Apply boundary conditions to the FD matrix in one dimension. """
function boundary_conditions_1d!(Dₓ, Dₓₓ, bc)
    if bc == :dirichlet # Dirichlet BC
        # bc1 = (-2.0*f[1] + f[2]) / h
        # bc2 = (f[end-1] - 2.0*f[end]) / h
        # return bc1, bc2
    elseif bc == :hobc # High order BC (Second order derivative)
        # length(f) > 4 || throw("Too few points input for this BC.")
        # c = [2.0, -5.0, 4.0, -1.0]
        # bc1 = sum(c .* f[1:4]) / h
        # bc2 = sum(reverse(c) .* f[end-3:end]) / h
        # return bc1, bc2
        Dₓₓ[1, 1:4] = [2.0 -5.0 4.0 -1.0]
        Dₓₓ[end, end-4:end] = [-1.0 4.0 -5.0 2.0]
    elseif bc == :periodic # Periodic BC
        # bc1 = (f[end] - 2.0*f[1] + f[2]) / h
        # bc2 = (f[end-1] - 2.0*f[end] + f[1]) / h
        # return bc1, bc2
        Dₓₓ[1, end] = 1.0
        Dₓₓ[end, 1] = 1.0
    elseif bc == :neumann # Neumann BC
        # bc1 = (f[2] - f[1]) / h # first order derivative
        # bc2 = (f[end] - f[end-1]) / h # first order derivative
        Dₓ[1, 1], Dₓ[1, 2] = -2.0, 2.0
        Dₓ[end, end-1], Dₓ[end, end] = -2.0, 2.0
        Dₓₓ[1, :] .= 0.0
        Dₓₓ[end, :] .= 0.0
    end
end

# ╔═╡ cf371b10-67ec-11eb-3343-853a9d0c234b
""" Build FD matrices in one dimension. """
function fd_der2_1d(Δ, c, nx, bc)
    Dₓ = spars.sparse(linalg.Tridiagonal(-ones(nx-1), zeros(nx), ones(nx-1)))
    Dₓₓ = spars.sparse(linalg.Tridiagonal(ones(nx-1), -2.0*ones(nx), ones(nx-1)))
    boundary_conditions_1d!(Dₓ, Dₓₓ, bc)
    Dₓ .*= (0.5/Δ)
    Dₓₓ ./= Δ^2
    return Dₓ, Dₓₓ
end

# ╔═╡ f0e7387e-67ec-11eb-3fd2-336fee2eb124
""" Build operator matrix with BCs in one dimension. """
function build_operators_1d(Dₓ, Dₓₓ, c, nx, b, fbc)
    A = c[1] .* Dₓₓ + c[2] .* Dₓ + c[3] .* sparse_identity(nx, nx)
    f = copy(b)
	# The right-hand side of the system is affected by the BC values 
	for i in 1:length(fbc.idx)
		f[fbc.idx[i]] = b[fbc.idx[i]] - fbc.val[i]
	end

    return A, f
end

# ╔═╡ f0cf1ca0-67ec-11eb-2cae-3df7f6ae7416
x = [0.0, 1.0] # region values

# ╔═╡ 14d2f8a0-67ee-11eb-0427-effe2e1a4916
u = [1.0, -1.0] # boundary values of u

# ╔═╡ 46647c40-67ee-11eb-1b0b-b11ca7b71d19
c = [1.0, 0.0, 0.0] # coefficients of the DE

# ╔═╡ f0b4dde0-67ec-11eb-3b78-63c700ef83f5
n = 41 # size of the grid

# ╔═╡ be97bc10-67ec-11eb-383f-cb142f204500
Δ = (x[2] - x[1]) / (n - 1) # step size, grid resolution	

# ╔═╡ c6064870-67ee-11eb-0aa2-572728b98eea
y = (0 : n-1) * Δ # axis 

# ╔═╡ 9288c160-67f0-11eb-270f-9bdb94123856
f(y) = -π^2 * cos(π * y) # source function

# ╔═╡ 8c8b4272-67f4-11eb-27d7-2fc0ff7b682a
# rhs of the linear system. idx field indicates the indices where the BC are, and val field it's the values that affects the BCs at each idx
fbc = (idx=[1, n], val=u./Δ^2) 

# ╔═╡ f19fd2c0-67ef-11eb-338f-4f2315d23fb3
uₐ = cos.(π.*y) # analytical solution

# ╔═╡ be848230-67ec-11eb-2db1-9d629d8d68fa
Dₓ, Dₓₓ = fd_der2_1d(Δ, c, n, :dirichlet)

# ╔═╡ be714850-67ec-11eb-3938-45e7696113de
A, b = build_operators_1d(Dₓ, Dₓₓ, c, n, f.(y), fbc)

# ╔═╡ be625430-67ec-11eb-1fa1-03b0f6742211
U = A \ Array(b)

# ╔═╡ be4aad80-67ec-11eb-09ff-87fbed74643f
plot(y, [U, uₐ], label=["FD" "Analytical"], title="Solution", xlabel="x")

# ╔═╡ be3773a0-67ec-11eb-2ce8-8194600b8a22
plot(y, linalg.norm.(U .- uₐ, Inf), lab="", title="Error", xlabel="x")

# ╔═╡ bdc9e4c0-67ec-11eb-0365-df7a5212c170


# ╔═╡ bdb8cdc2-67ec-11eb-1db1-e55dc16d8554


# ╔═╡ bda56cd0-67ec-11eb-10c9-0d0a847073ba


# ╔═╡ bd96c6d0-67ec-11eb-0077-5f217cd7ced9


# ╔═╡ bd8058a0-67ec-11eb-0596-75f8bec6e72b


# ╔═╡ bd699c50-67ec-11eb-198f-41ce40021691


# ╔═╡ bd4f3680-67ec-11eb-387f-3158ff4abdee


# ╔═╡ bd36cc82-67ec-11eb-25e8-0b47f048ec29


# ╔═╡ bd086980-67ec-11eb-31b1-799a7d623102


# ╔═╡ bd06e2e0-67ec-11eb-14fd-010d9f31ccd3


# ╔═╡ bd04c000-67ec-11eb-2fd4-593943a6d60d


# ╔═╡ bd02eb40-67ec-11eb-10bc-6d0a6cd44eac


# ╔═╡ bcb196f0-67ec-11eb-00a4-517e2df3f70e


# ╔═╡ Cell order:
# ╟─f5ac2580-67ea-11eb-1e9c-272c12b93242
# ╠═cfc2b940-67ec-11eb-07e6-2d8f575d0e14
# ╠═cfabaed0-67ec-11eb-2dfd-c3c6cdf2b692
# ╠═cf94f280-67ec-11eb-14e0-dfefde71afaa
# ╠═cf7a8cb0-67ec-11eb-2e7b-83f9241fbe11
# ╠═cf672bc0-67ec-11eb-2ab0-2bd22cafff34
# ╠═cf5180e0-67ec-11eb-1cfd-1d3026c131c8
# ╠═cf371b10-67ec-11eb-3343-853a9d0c234b
# ╠═f0e7387e-67ec-11eb-3fd2-336fee2eb124
# ╠═f0cf1ca0-67ec-11eb-2cae-3df7f6ae7416
# ╠═14d2f8a0-67ee-11eb-0427-effe2e1a4916
# ╠═46647c40-67ee-11eb-1b0b-b11ca7b71d19
# ╠═f0b4dde0-67ec-11eb-3b78-63c700ef83f5
# ╠═be97bc10-67ec-11eb-383f-cb142f204500
# ╠═c6064870-67ee-11eb-0aa2-572728b98eea
# ╠═9288c160-67f0-11eb-270f-9bdb94123856
# ╠═8c8b4272-67f4-11eb-27d7-2fc0ff7b682a
# ╠═f19fd2c0-67ef-11eb-338f-4f2315d23fb3
# ╠═be848230-67ec-11eb-2db1-9d629d8d68fa
# ╠═be714850-67ec-11eb-3938-45e7696113de
# ╠═be625430-67ec-11eb-1fa1-03b0f6742211
# ╠═be4aad80-67ec-11eb-09ff-87fbed74643f
# ╠═be3773a0-67ec-11eb-2ce8-8194600b8a22
# ╠═bdc9e4c0-67ec-11eb-0365-df7a5212c170
# ╠═bdb8cdc2-67ec-11eb-1db1-e55dc16d8554
# ╠═bda56cd0-67ec-11eb-10c9-0d0a847073ba
# ╠═bd96c6d0-67ec-11eb-0077-5f217cd7ced9
# ╠═bd8058a0-67ec-11eb-0596-75f8bec6e72b
# ╠═bd699c50-67ec-11eb-198f-41ce40021691
# ╠═bd4f3680-67ec-11eb-387f-3158ff4abdee
# ╠═bd36cc82-67ec-11eb-25e8-0b47f048ec29
# ╠═bd086980-67ec-11eb-31b1-799a7d623102
# ╠═bd06e2e0-67ec-11eb-14fd-010d9f31ccd3
# ╠═bd04c000-67ec-11eb-2fd4-593943a6d60d
# ╠═bd02eb40-67ec-11eb-10bc-6d0a6cd44eac
# ╠═bcb196f0-67ec-11eb-00a4-517e2df3f70e
