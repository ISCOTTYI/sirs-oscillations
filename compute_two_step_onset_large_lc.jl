using StatsBase
include("utils.jl")
include("erlang_two_step_kernel.jl")

β, ρ = 2, 1
T_hopf = 43.5
Ts = 25.5:0.5:T_hopf
N_crits = []
for T in Ts
    print(T)
    flush(stdout)
    fp = erlang_two_step_kernel_fp(β, ρ, T)
    found = false
    N = 6
    while !found
        sol = solve_erlang_two_step_kernel(β, ρ, T, N, 0.01, 6000)
        if abs(mean(sol[1, end-100:end]) - fp) > 1e-5
            found = true
            push!(N_crits, N)
        end
        N += 2
    end
end

writedlm_header(
    "./data/two-step-kernel/large-lc-death.dat",
    [Ts, N_crits],
    "x: Time delay constants T, all < T_hopf; y: Critical N values at which large limit cycle is born -> multistability, β = $β, ρ = $ρ"
)