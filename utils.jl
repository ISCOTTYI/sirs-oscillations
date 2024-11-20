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

function writedlm_header(filename::AbstractString, data::AbstractArray, header::AbstractString; kwargs...)
    open(filename, "w") do io::IOStream
        write(io, "# $header\n")
        writedlm(io, data, kwargs...)
    end
end

function save_savefig(fig, path; png = false, svg = false, overwrite_png = false, dpi=500, kwargs...)
    if png
        if !overwrite_png && isfile(path * ".png")
            println("png file already exists. Skipping.")
            flush(stdout)
        else
            fig.savefig(path * ".png", dpi=dpi, kwargs...)
        end
    end
    if svg
        if isfile(path * ".svg")
            println("svg file already exists. Skipping.")
            flush(stdout)
            return
        else
            fig.savefig(path * ".svg", kwargs...)
        end
    end
end
