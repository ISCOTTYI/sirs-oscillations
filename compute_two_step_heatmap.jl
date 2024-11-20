using DelimitedFiles

include("utils.jl")
include("erlang_two_step_kernel.jl")

# Define parameters
β, ρ = 2, 1
Ns = 2:4:500
Ts = 25:0.1:70

curr_N = Ns[1]
function computation(β, ρ, T, N)
    if N != curr_N
        println(N)
        global curr_N = N
    end
    return λ_max(erlang_two_step_kernel_jacobian(β, ρ, T, N))
end

# Compute heatmap
z = [computation(β, ρ, T, N) for T in Ts, N in Ns]

# Save data
writedlm_header(
    "./data/two-step-kernel/new-oscillation-onset-heatmap.dat",
    z,
    "β = $β, ρ = $ρ, Ns = $Ns, Ts = $Ts"
)
