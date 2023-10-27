import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def SIR_demography(y, t, beta, gamma, Lambda, mu):
    S, I, R = y
    dS_dt = Lambda - beta * S * I / (S + I + R) - mu * S
    dI_dt = beta * S * I / (S + I + R) - gamma * I - mu * I
    dR_dt = gamma * I - mu * R
    return [dS_dt, dI_dt, dR_dt]

def GA_with_demography(S0, I0, R0, beta, gamma, Lambda, mu, max_time):
    S, I, R = S0, I0, R0
    t = 0
    times = [t]
    S_values = [S]
    I_values = [I]
    R_values = [R]

    while t < max_time and I > 0:
        N = S + I + R

        # Calculate propensities
        a1 = beta * S * I / N        # Transmission
        a2 = gamma * I               # Recovery
        a3 = Lambda                  # Birth
        a4 = mu * S                  # Death of a susceptible
        a5 = mu * I                  # Death of an infected
        a6 = mu * R                  # Death of a recovered

        a0 = a1 + a2 + a3 + a4 + a5 + a6

        # Time until next event
        dt = -np.log(np.random.random()) / a0
        t += dt

        # Determine which event occurs
        r = np.random.random() * a0
        if r < a1:
            # Transmission event
            S -= 1
            I += 1
        elif r < a1 + a2:
            # Recovery event
            I -= 1
            R += 1
        elif r < a1 + a2 + a3:
            # Birth event
            S += 1
        elif r < a1 + a2 + a3 + a4:
            # Death of a susceptible
            S -= 1
        elif r < a1 + a2 + a3 + a4 + a5:
            # Death of an infected
            I -= 1
        else:
            # Death of a recovered
            R -= 1

        # Store results
        times.append(t)
        S_values.append(S)
        I_values.append(I)
        R_values.append(R)

    return times, S_values, I_values, R_values

# Parameters
I0 = 10
R0 = 0
Lambda = 0.05  # birth rate
mu = 0.05  # death rate
max_time = 70
t = np.linspace(0, max_time, 1000)

# Parameters for the experiment
R_0_values = [1.5, 3, 5]
N_values = [100, 500, 1000]
gamma = 0.1
num_runs = 100

# Storage for results
results = []

for R_0 in R_0_values:
    for N in N_values:
        S0 = N - I0 - R0
        beta = R_0 * (gamma + mu)

        # Run stochastic simulations multiple times
        all_S = []
        all_I = []
        end_times = []
        covariances = []
        for _ in range(num_runs):
            times, S_values, I_values, _ = GA_with_demography(S0, I0, R0, beta, gamma, Lambda, mu, max_time)

            # Interpolate onto the common time grid
            S_interp = np.interp(t, times, S_values)
            I_interp = np.interp(t, times, I_values)

            below_I0_indices = np.where(I_interp <= I0-0.01)[0]

            if below_I0_indices.size > 0:
                end_index = below_I0_indices[0]
                end_time = t[end_index]

                # Calculate covariance for this run up to the end time
                cov_SI = np.cov(S_interp[:end_index + 1], I_interp[:end_index + 1])[0, 1]
                covariances.append(cov_SI)

            all_S.append(S_interp)
            all_I.append(I_interp)

        if not covariances:  # No valid simulations with I reaching I0-1
            continue


    


        # Covariance experiment
        start_time = 0
        start_index = np.where(t >= start_time)[0][0]
        end_index = np.where(t <= end_time)[0][-1] + 1
    
        mean_S = np.mean(all_S, axis=0)
        mean_I = np.mean(all_I, axis=0)
        mean_S_mod = np.mean([s[start_index:end_index] for s in all_S], axis=0)
        mean_I_mod = np.mean([i[start_index:end_index] for i in all_I], axis=0)
        std_S = np.std(mean_S[start_index:end_index])
        std_I = np.std(mean_I[start_index:end_index])
        covariance_SI = np.cov(mean_S_mod, mean_I_mod)[0, 1]
        corr_coeff = np.corrcoef(mean_S_mod, mean_I_mod)[0, 1] 
        corr_coefficient = covariance_SI / (std_S * std_I)

        # Run deterministic model
        solution = odeint(SIR_demography, [S0, I0, R0], t, args=(beta, gamma, Lambda, mu))
        S_det, I_det, R_det = solution.T

        # Store results
        results.append({
            'R_0': R_0,
            'N': N,
            'mean_S': mean_S,
            'mean_I': mean_I,
            'covariance_SI': covariance_SI,
            'corr_coefficient': corr_coefficient,
            'corr_coeff': corr_coeff,
            'S_det': S_det,
            'I_det': I_det
        })

# Create a grid of subplots
num_R_0 = len(R_0_values)
num_N = len(N_values)


# Create a grid of subplots
fig, axes = plt.subplots(num_R_0, num_N, figsize=(15, 15), sharex=True)

for i, res in enumerate(results):
    row = i // num_N
    col = i % num_N
    
    ax = axes[row, col]
    
    # Get the N value for this subplot
    N = res['N']
    
    # Recalculate initial conditions based on N
    S0 = N - I0 - R0
    
    # Plot on the current subplot
    ax.plot(t, res['mean_S'], label="Mean Susceptible (Stochastic)", color="orange")
    ax.plot(t, res['mean_I'], label="Mean Infectious (Stochastic)", color="red")
    ax.plot(t, res['S_det'], label="Susceptible (Deterministic)", color="orange", linestyle="--")
    ax.plot(t, res['I_det'], label="Infectious (Deterministic)", color="red", linestyle="--")
    
    # Include initial conditions in the plot title
    ax.set_title(f"R₀={res['R_0']}, N={N}, S₀={S0}, I₀={I0}, R₀={R0}\nCorrelation coefficient (S,I)={res['corr_coeff']:.2f}")
    
    ax.set_ylim(0, N)  # Set y-axis limit based on N
    
    if row == num_R_0 - 1:
        ax.set_xlabel("Time")
    if col == 0:
        ax.set_ylabel("Population")
    if i == 0:
        ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()


