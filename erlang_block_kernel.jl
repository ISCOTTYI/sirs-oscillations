# System defined explicitly for performance reasons, despite everything can be
# derived from general kernel superposition
include("erlang_superpos_kernel.jl")

# Expansion coefficients
block_kernel_cₘ(m::Integer) = 1

# Endemic state for the infected
block_kernel_I_end(β::Real, ρ::Real, T::Real) = (β - ρ) / (β * (T * ρ + 1))

function erlang_block_kernel!(
    du::Vector{<:Real}, u::Vector{<:Real}, p, t::Real
)
    """
    System definition for the SIRS model with block kernel for dwell-time in
    recovered compartment given as superposition of Erlang kernels.
    Equivalent to model by (Hethcote, 1981).
        R = ∑ₘ Rₘ
    Dimensionality: N + 1
    Variables: Infected I at [1], auxiliary recovered R_m at [m+1 ∈ [2, N+1]]
    Normalization: 1 = S + I + R
    """
    β, ρ, T, N = p
    du[1] = β * u[1] * (1 - sum(u)) - ρ * u[1]
    du[2] = ρ * u[1] - (N/T) * u[2]
    for m in 3:N+1
        du[m] = N * (u[m-1] - u[m]) / T
    end
end

function solve_erlang_block_kernel(
        β::Real, ρ::Real, T::Real,
        N::Integer, I₀::Real, t_max::Real
    )
    """
    Solves the SIRS model with block kernel for dwell-time in
    recovered compartment as superposition of Erlang kernels.
    Equivalent to model by (Hethcote, 1981).
        R = ∑ₘ Rₘ
    Dimensionality: N + 1
    Variables: Infected I at [1], auxiliary recovered R_m at [m+1 ∈ [2, N+1]]
    Normalization: 1 = S + I + R
    """
    p = (β, ρ, T, N)
    u₀ = zeros(N+1); u₀[1] = I₀
    sol = solve(
        ODEProblem(erlang_block_kernel!, u₀, (0.0, t_max), p),
        RK4(),
        # alg_hints = [:stiff],
        reltol = 1e-8
    )
    return sol
end

function erlang_block_kernel_jacobian(
        β::Real, ρ::Real, T::Real, N::Integer
    )
    """
    Generates and returns the Jacobian matrix of the SIRS model with
    block delay as a sum of Erlang kernels evaluated at the non-trivial
    fixpoint
        I = (β - ρ) / (β (T ρ + 1)),
        Rₘ = (ρ T / N) I
    """
    dim::Integer = N + 1
    jac::Matrix{Float64} = diagm(0 => fill(-N/T, dim), -1 => fill(N/T, dim-1))
    jac[1, 1] = (-β + ρ) / (T * ρ + 1)
    jac[1, 2:end] .= (-β + ρ) / (T * ρ + 1)
    jac[2, 1] = ρ
    return jac
end

function erlang_block_kernel_oscillation_onset(
        β::Real, ρ::Real, T::Real;
        N_max::Integer = 1000
    )
    N = 2
    while N < N_max
        if λ_max(erlang_block_kernel_jacobian(β, ρ, T, N)) > 0
            return N
        end
        N += 1
    end
    @error "No positive eigenvalue found. Maybe increase N_max?"
    return N_max 
end
