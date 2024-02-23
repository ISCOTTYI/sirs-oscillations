using DifferentialEquations
using LinearAlgebra

function erlang_sum_kernel!(
    du::Vector{<:Real}, u::Vector{<:Real}, p, t::Real
)
"""
System definition for the SIRS model with block delay as a sum of
Erlang kernels (kernel series framework). Model from (Hethcote, 1981).
Dimensionality: N + 1
Variables: Infected I at [1], auxiliary recovered R_m at [m+1 ∈ [2, N+1]]
Details:
    * R_1 + ... + R_N = R
    * 1 = S + I + R
"""
β, ρ, T, N = p
du[1] = β * u[1] * (1 - sum(u)) - ρ * u[1]
du[2] = ρ * u[1] - (N/T) * u[2]
for m in 3:N+1
    du[m] = N * (u[m-1] - u[m]) / T
end
end

erlang_sum_kernel_R(sol) = vec(sum(sol[2:end, :]; dims = 1))
erlang_sum_kernel_S(sol) = 1 - I - erlang_sum_kernel_R(sol)

function solve_erlang_sum_kernel(
    β::Real, ρ::Real, T::Real,
    N::Integer, I₀::Real, t_max::Real
)
p = (β, ρ, T, N)
u₀ = zeros(N+1); u₀[1] = I₀
sol = solve(
    ODEProblem(erlang_sum_kernel!, u₀, (0.0, t_max), p),
    RK4(),
    # alg_hints = [:stiff],
    reltol = 1e-8
)
return sol
end

function erlang_sum_kernel_jacobian(
    β::Real, ρ::Real,
    T::Real, N::Integer
)
"""
Generates and returns the Jacobian matrix of the SIRS model with
block delay as a sum of Erlang kernels evaluated at the non-trivial
fixpoint
    I = (β - ρ) / (β (T ρ + 1)),
    Rₘ = (ρ T / N) I
"""
dim::Integer = N+1
jac::Matrix{Float64} = diagm(0 => fill(-N/T, dim), -1 => fill(N/T, dim-1))
jac[1, 1] = (-β + ρ)/(T*ρ + 1)
jac[1, 2:end] .= (-β + ρ)/(T*ρ + 1)
jac[2, 1] = ρ
return jac
end

function maximum_eigenvalue(M::Matrix)
"""
Utility function - returns maximum of the matrix M, judged by real part.
"""
return argmax(real, eigvals(M))
end

_λ_max(β, ρ, T, N) = real(maximum_eigenvalue(erlang_sum_kernel_jacobian(β, ρ, T, N)))
