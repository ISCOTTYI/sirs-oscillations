using DifferentialEquations
using StatsBase
using LinearAlgebra

function maximum_eigenvalue(M::Matrix)
    """
    Utility function - returns maximum of the matrix M, judged by real part.
    """
    return argmax(real, eigvals(M))
end

λ_max(J::Matrix) = real(maximum_eigenvalue(J))

function normalized_sol_hist(
        sol::ODESolution, sol_idx::Integer, t_trans::Real, t_max::Real;
        Δt::Real = 0.01, nbins::Integer = 100
    )
    sample_ts = t_trans:Δt:t_max
    hist = fit(Histogram, sol(sample_ts, idxs = sol_idx).u, nbins = nbins)
    hist = normalize(hist, mode = :probability)
    return hist
end

function pearson_skew(
        sol::ODESolution, sol_idx::Integer, t_trans::Real, t_max::Real;
        Δt::Real = 0.01
    )
    sample_ts = t_trans:Δt:t_max
    time_series = sol(sample_ts, idxs = sol_idx).u
    return (mean(time_series) - median(time_series)) / std(time_series)
end

function skew(
        sol::ODESolution, sol_idx::Integer, t_trans::Real, t_max::Real;
        Δt::Real = 0.01
    )
    sample_ts = t_trans:Δt:t_max
    time_series = sol(sample_ts, idxs = sol_idx).u
    return skewness(time_series)
end
