using Distributions

# Step function kernel
Θ(τ, T) = 0 <= τ <= T ? 1 : 0;
# Regular Erlang Kernel
Kₘ(τ, Tₙ, m) = pdf(Erlang(m, Tₙ), τ);
# Erlang sum kernel
Θ_ker(τ, T, N) = T/N * sum([Kₘ(τ, T/N, m) for m in 1:N]);
# Erlang sum kernel in terms of inc. gamma function (gamma_inc[2] = Γ/(N-1)!)
Θ_ker_Γ(τ, T, N) = gamma_inc(N, N * τ / T)[2];
# Maximum decline
sₙ(T, N) = - Kₘ(T*(N-1) / N, T/N, N);