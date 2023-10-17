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

# Parameters for the experiment
beta_values = [0.2, 0.3, 0.4]
gamma_values = [0.05, 0.1, 0.15]
num_runs = 100

# Storage for results
results = []

# Define a common time grid
t_common = np.linspace(0, max_time, 1000)

for beta in beta_values:
    for gamma in gamma_values:
        # Run stochastic simulations multiple times
        all_S = []
        all_I = []
        for _ in range(num_runs):
            times, S_values, I_values, _ = GA_with_demography(S0, I0, R0, beta, gamma, Lambda, mu, max_time)
            
            # Interpolate onto the common time grid
            S_interp = np.interp(t_common, times, S_values)
            I_interp = np.interp(t_common, times, I_values)
            
            all_S.append(S_interp)
            all_I.append(I_interp)

        # Compute mean and covariance
        mean_S = np.mean(all_S, axis=0)
        mean_I = np.mean(all_I, axis=0)
        covariance_SI = np.cov(mean_S, mean_I)[0, 1]

        # Run deterministic model
        solution = odeint(SIR_demography, [S0, I0, R0], t, args=(beta, gamma, Lambda, mu))
        S_det, I_det, _ = solution.T

        # Store results
        results.append({
            'beta': beta,
            'gamma': gamma,
            'mean_S': mean_S,
            'mean_I': mean_I,
            'covariance_SI': covariance_SI,
            'S_det': S_det,
            'I_det': I_det
        })

# Create a grid of subplots
num_beta = len(beta_values)
num_gamma = len(gamma_values)
fig, axes = plt.subplots(num_beta, num_gamma, figsize=(15, 15))

first_plot = True

for i, res in enumerate(results):
    ax = axes[i // num_gamma][i % num_gamma]
    
    # Plot on the current subplot
    ax.plot(t_common, res['mean_S'], label="Mean Susceptible (Stochastic)" if first_plot else "", color="orange")
    ax.plot(t_common, res['mean_I'], label="Mean Infectious (Stochastic)" if first_plot else "", color="red")
    ax.plot(t_common, res['S_det'], label="Susceptible (Deterministic)" if first_plot else "", color="orange", linestyle="--")
    ax.plot(t_common, res['I_det'], label="Infectious (Deterministic)" if first_plot else "", color="red", linestyle="--")
    
    # Add recovered individuals to the plot
    mean_R = np.array(S0) - np.array(res['mean_S']) - np.array(res['mean_I'])  # Calculate mean R from S and I
    R_det = solution[:, 2]  # Extract R values from deterministic solution
    ax.plot(t_common, mean_R, label="Mean Recovered (Stochastic)" if first_plot else "", color="green")
    ax.plot(t_common, R_det, label="Recovered (Deterministic)" if first_plot else "", color="green", linestyle="--")
    
    ax.set_title(f"β={res['beta']}, γ={res['gamma']}\nCovariance(S,I)={res['covariance_SI']:.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.grid(True)
    
    first_plot = False

# Adjust layout to prevent overlap and add more space between plots
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.5)

# Create the legend at the bottom
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.2))

plt.show()

