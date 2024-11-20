using DifferentialEquations
using LinearAlgebra

include("utils.jl")

function erlang_superpos_kernel!(
        du::Vector{<:Real}, u::Vector{<:Real}, p, t::Real
    )
    """
    System definition for the SIRS model with kernel for dwell-time in
    recovered compartment as general superposition of Erlang kernels.
    Derived from Model by (Hethcote, 1981).
        R = ∑ₘ cₘ Rₘ
    Dimensionality: N + 1
    Variables: Infected I at [1], auxiliary recovered R_m at [m+1 ∈ [2, N+1]]
    Normalization: 1 = S + I + R
    """
    β, ρ, T, N, cₘ = p
    R = sum([cₘ(m-1) * u[m] for m in 2:N+1])
    du[1] = β * u[1] * (1 - u[1] - R) - ρ * u[1]
    du[2] = ρ * u[1] - (N/T) * u[2]
    for m in 3:N+1
        du[m] = N * (u[m-1] - u[m]) / T
    end
end

# TODO!
# erlang_superpos_kernel_R(sol, cₘ) = vec(sum(sol[2:end, :]; dims = 1))
# erlang_superpos_kernel_S(sol, cₘ) = 1 - sol[1, :] - erlang_superpos_kernel_R(sol, cₘ)

function solve_erlang_superpos_kernel(
        β::Real, ρ::Real, T::Real,
        N::Integer, cₘ::Function,
        I₀::Real, t_max::Real
    )
    p = (β, ρ, T, N, cₘ)
    u₀ = zeros(N+1); u₀[1] = I₀
    sol = solve(
        ODEProblem(erlang_superpos_kernel!, u₀, (0.0, t_max), p),
        RK4(),
        # alg_hints = [:stiff],
        reltol = 1e-8
    )
    return sol
end

function erlang_superpos_kernel_jacobian(
        β::Real, ρ::Real, T::Real,
        N::Integer, cₘ::Function,
        I_endemic::Real
    )
    dim::Integer = N + 1
    Rₘ_endemic::Real = T * ρ * I_endemic / N
    R_endemic::Real = sum([cₘ(m) * Rₘ_endemic for m in 1:N])
    jac::Matrix{Float64} = diagm(0 => fill(-N/T, dim), -1 => fill(N/T, dim-1))
    jac[1, 1] = β * (1 - 2 * I_endemic - R_endemic) - ρ
    jac[1, 2:end] = [-β * I_endemic * cₘ(m) for m in 1:N]
    jac[2, 1] = ρ
    return jac
end

# # FP for system with c(m) = m <= N_mid ? k₁ : k₂
# general_two_step_kernel_fp(β, ρ, T, N, N_mid, k₁, k₂) = N * (β - ρ) / (β * (N * T * k₂ * ρ + N + N_mid * T * k₁ * ρ - N_mid * T * k₂ * ρ))
