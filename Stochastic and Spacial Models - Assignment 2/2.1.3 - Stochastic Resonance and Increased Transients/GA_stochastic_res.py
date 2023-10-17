import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Deterministic SIR model
def SIR_demography(y, t, beta, gamma, Lambda, mu):
    S, I, R = y
    dS_dt = Lambda - beta * S * I / (S + I + R) - mu * S
    dI_dt = beta * S * I / (S + I + R) - gamma * I - mu * I
    dR_dt = gamma * I - mu * R
    return [dS_dt, dI_dt, dR_dt]

def GA_with_demography(S0, I0, R0, beta, gamma, Lambda, mu, max_time):
    # Initial conditions
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
S0= 990
I0=10
R0=0
beta = 0.3
gamma = 0.1
Lambda = 5  # birth rate
mu = 0.01  # death rate
max_time = 200
t = np.linspace(0, max_time, 1000)

# Define a function to measure the transient (deviation from deterministic equilibrium)
def measure_transient(stochastic, deterministic):
    return np.max(np.abs(stochastic - deterministic))

# Parameters
S0 = 990
I0 = 10
R0 = 0
beta = 0.3
gamma = 0.1
max_time = 160
t = np.linspace(0, max_time, 1000)



# Parameters for the experiment
N_values = [100, 500, 1000, 5000]
beta_values = [0.1, 0.2, 0.3, 0.4]
gamma = 0.1  # Fixed recovery rate for simplicity
max_time = 160
t_common = np.linspace(0, max_time, 1000)

transients = []

for N in N_values:
    for beta in beta_values:
        # Adjust initial conditions based on N
        I0 = 5
        S0 = N - I0 - R0
        
        # Run stochastic simulation
        times, S_stoch, I_stoch, R_stoch = GA_with_demography(S0, I0, R0, beta, gamma, Lambda, mu, max_time)
        
        # Interpolate stochastic results onto common time grid
        S_stoch_interp = np.interp(t_common, times, S_stoch)
        I_stoch_interp = np.interp(t_common, times, I_stoch)
        
        # Solve ODE for deterministic SIR
        solution = odeint(SIR_demography, [S0, I0, R0], t, args=(beta, gamma, Lambda, mu))
        S_det, I_det, R_det = solution.T
        
        # Measure transients
        transient_S = measure_transient(S_stoch_interp, S_det)
        transient_I = measure_transient(I_stoch_interp, I_det)
        
        transients.append({
            'N': N,
            'beta': beta,
            'transient_S': transient_S,
            'transient_I': transient_I,
            'S_stoch': S_stoch_interp,
            'I_stoch': I_stoch_interp,
            'S_det': S_det,
            'I_det': I_det
        })

# Plotting
fig, axes = plt.subplots(len(N_values), len(beta_values), figsize=(15, 15), sharex=True, sharey=True)

for i, res in enumerate(transients):
    ax = axes[i // len(beta_values)][i % len(beta_values)]
    
    # Plot on the current subplot
    ax.plot(t_common, res['S_stoch'], label="Susceptible (Stochastic)", color="orange")
    ax.plot(t_common, res['I_stoch'], label="Infectious (Stochastic)", color="red")
    ax.plot(t_common, res['S_det'], label="Susceptible (Deterministic)", color="orange", linestyle="--")
    ax.plot(t_common, res['I_det'], label="Infectious (Deterministic)", color="red", linestyle="--")
    
    ax.set_title(f"N={res['N']}, β={res['beta']}\nTransient S = {res['transient_S']:.2f}, I = {res['transient_I']:.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.legend()
    ax.grid(True)

# Adjust layout to prevent overlap and add more space between plots
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()
