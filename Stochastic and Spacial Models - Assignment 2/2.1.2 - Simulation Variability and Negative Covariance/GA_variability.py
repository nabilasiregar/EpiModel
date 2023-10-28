import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pdb

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
I0= 10
R0= 0
Lambda = 0.05  # birth rate
mu = 0.05  # death rate
max_time = 70
t = np.linspace(0, max_time, 1000)

# Parameters for the experiment
beta_values = [0.8, 0.9, 0.99]
gamma_values = [0.05, 0.1, 0.3]
num_runs = 100

# Storage for results
results = []


for beta in beta_values:
    for gamma in gamma_values:
        # Run stochastic simulations multiple times
        all_S = []
        all_I = []
        end_times = []
        for _ in range(num_runs):
            times, S_values, I_values, _ = GA_with_demography(S0, I0, R0, beta, gamma, Lambda, mu, max_time)
            
            # Interpolate onto the common time grid
            S_interp = np.interp(t, times, S_values)
            I_interp = np.interp(t, times, I_values)
            
            below_10_percent = np.where(I_interp <= 0.1 * I0)[0]
            if below_10_percent.size > 0:
                end_time = t[below_10_percent[0]]
                end_times.append(end_time)

            all_S.append(S_interp)
            all_I.append(I_interp)

        if not end_times:
            continue

        end_time=max(end_times)

        # Covariance experiment
        start_time = 0
        start_index = np.where(t >= start_time)[0][0]
        end_index = np.where(t <= end_time)[0][-1] + 1

        # Compute mean and covariance
        mean_S = np.mean(all_S, axis=0)
        mean_I = np.mean(all_I, axis=0)
        mean_S_mod = np.mean([s[start_index:end_index] for s in all_S], axis=0)
        mean_I_mod = np.mean([i[start_index:end_index] for i in all_I], axis=0)
        std_S = np.std(S_values, axis=0)
        std_I = np.std(I_values, axis=0)
        covariance_SI = np.cov(mean_S_mod, mean_I_mod)[0, 1]
        corr_coefficient = covariance_SI / (std_S * std_I)

        # Run deterministic model
        solution = odeint(SIR_demography, [S0, I0, R0], t, args=(beta, gamma, Lambda, mu))
        S_det, I_det, R_det = solution.T

        # Store results
        results.append({
            'beta': beta,
            'gamma': gamma,
            'mean_S': mean_S,
            'mean_I': mean_I,
            'covariance_SI': covariance_SI,
            'corr_coefficient': corr_coefficient,
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
    ax.plot(t, res['mean_S'], label="Mean Susceptible (Stochastic)" if first_plot else "", color="orange")
    ax.plot(t, res['mean_I'], label="Mean Infectious (Stochastic)" if first_plot else "", color="red")
    ax.plot(t, res['S_det'], label="Susceptible (Deterministic)" if first_plot else "", color="orange", linestyle="--")
    ax.plot(t, res['I_det'], label="Infectious (Deterministic)" if first_plot else "", color="red", linestyle="--")
    
    # # Add recovered individuals to the plot
    # mean_R = np.array(S0) + np.cumsum(Lambda - mu*(res['mean_S'] + res['mean_I']))*np.diff(t)[0] - np.array(res['mean_S']) - np.array(res['mean_I'])
    # R_det = solution[:, 2]
    # ax.plot(t, mean_R, label="Mean Recovered (Stochastic)" if first_plot else "", color="green")
    # ax.plot(t, R_det, label="Recovered (Deterministic)" if first_plot else "", color="green", linestyle="--")
    
    ax.set_title(f"β={res['beta']}, γ={res['gamma']}\nCorrelation coefficient (S,I)={res['corr_coefficient']:.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.grid(True)
    
    
    first_plot = False

# Plot settings
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.5)

# Create the legend at the bottom
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.2))

plt.show()

