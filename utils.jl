using DifferentialEquations
using StatsBase
using LinearAlgebra

function normalized_sol_hist(
    sol::ODESolution, sol_idx::Integer, t_trans::Real, t_max::Real;
    Δt::Real = 0.01, nbins::Integer = 100
)
sample_ts = t_trans:Δt:t_max
hist = fit(Histogram, sol(sample_ts, idxs = sol_idx).u, nbins = nbins)
hist = normalize(hist, mode = :probability)
return hist
end