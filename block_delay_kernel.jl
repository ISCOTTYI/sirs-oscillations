using DifferentialEquations
using LinearAlgebra

function step_function_kernel!(
        du::Vector{<:Real}, u::Vector{<:Real},
        h, p, t::Real
    )
    """
    System definition for the SIRS model with time delay as presented by
    (Hethcote et al., 1981) with a step function block kernel.
    Dimensionality: 2
    Variables: Infected I at [1] and recovered R at [2]
    """
    β, ρ, T = p
    du[1] = β * u[1] * (1 - u[1] - u[2]) - ρ * u[1]
    du[2] = ρ * (u[1] - h(p, t-T)[1])
end

function solve_step_function_kernel(
        β::Real, ρ::Real, T::Real, I₀::Real, t_max::Real
    )
    p = (β, ρ, T)
    h(p, t) = t == 0 ? [I₀, 0] : [0.0, 0.0] # infection seeded at t = 0
    sol = solve(
        DDEProblem(step_function_kernel!, [I₀, 0.0], h, (0.0, t_max), p),
        MethodOfSteps(RK4()),
        reltol = 1e-8
    )
    return sol
end
