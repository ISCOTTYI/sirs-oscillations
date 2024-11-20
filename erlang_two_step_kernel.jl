# System defined explicitly for performance reasons, despite everything can be
# derived from general kernel superposition
include("erlang_superpos_kernel.jl")

# Expansion coefficients
erlang_two_step_kernel_cₘ(m::Integer, N) = m <= N/2 ? 1 : 0.5

function erlang_two_step_kernel!(
        du::Vector{<:Real}, u::Vector{<:Real}, p, t::Real
    )
    """
    
    """
    β, ρ, T, N = p
    N₁::Integer = N/2 + 1
    N₂::Integer = N₁ + 1
    du[1] = β * u[1] * (1 - u[1] - sum(u[2:N₁]) - 1/2 * sum(u[N₂:end])) - ρ * u[1]
    du[2] = ρ * u[1] - (N/T) * u[2]
    for m in 3:N+1
        du[m] = N * (u[m-1] - u[m]) / T
    end
end

function solve_erlang_two_step_kernel(
        β::Real, ρ::Real, T::Real,
        N::Integer, I₀::Real, t_max::Real
    )
    """
    
    """
    p = (β, ρ, T, N)
    u₀ = zeros(N+1); u₀[1] = I₀
    sol = solve(
        ODEProblem(erlang_two_step_kernel!, u₀, (0.0, t_max), p),
        Tsit5(),
        # alg_hints = [:stiff],
        reltol = 1e-8
    )
    return sol
end

erlang_two_step_kernel_R(sol, N::Integer) = (sum(sol[2:Int(N/2+1), :], dims = 1) + sum(sol[Int(N/2+2):end, :] .* 0.5, dims = 1))[1, :]

function erlang_two_step_kernel_jacobian(
        β::Real, ρ::Real, T::Real, N::Integer
    )
    dim::Integer = N + 1
    jac::Matrix{Float64} = diagm(0 => fill(-N/T, dim), -1 => fill(N/T, dim-1))
    jac[1, 1] = - 4 * (β - ρ) / (3 * T * ρ + 4)
    N₁::Integer = N/2 + 1
    N₂::Integer = N₁ + 1
    jac[1, 2:N₁] .= (4 * (-β + ρ) / (3 * T * ρ + 4))
    jac[1, N₂:end] .= (2 * (-β + ρ) / (3 * T * ρ + 4))
    jac[2, 1] = ρ
    return jac
end

# FP for cₘ(m) = m <= N/2 ? 1 : 0.5
erlang_two_step_kernel_fp(β, ρ, T) = (1 - ρ / β) / (1 + 3/4 * ρ * T)
