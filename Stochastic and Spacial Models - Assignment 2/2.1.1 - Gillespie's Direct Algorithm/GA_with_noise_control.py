import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Deterministic SIR model
def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I / (S + I + R)
    dI_dt = beta * S * I / (S + I + R) - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]

def gillespie_SIR_scaled(S0, I0, R0, beta, gamma, max_time, noise_scale=1):
    # Initial conditions
    S, I, R = S0, I0, R0
    t = 0
    times = [t]
    S_values = [S]
    I_values = [I]
    R_values = [R]

    while t < max_time and I > 0:
        N = S + I + R

        # Calculate propensities with noise scaling
        a1 = noise_scale * beta * S * I / N
        a2 = noise_scale * gamma * I
        a0 = a1 + a2

        # Time until next event
        dt = -np.log(np.random.random()) / a0
        t += dt

        # Determine which event occurs
        r = np.random.random()
        if r < a1 / a0:
            # Transmission event
            S -= 1
            I += 1
        else:
            # Recovery event
            I -= 1
            R += 1

        # Store results
        times.append(t)
        S_values.append(S)
        I_values.append(I)
        R_values.append(R)

    return times, S_values, I_values, R_values


# Parameters
S0 = 990
I0 = 10
R0 = 0
beta = 0.3
gamma = 0.1
max_time = 200
t = np.linspace(0, max_time, 1000)

# Solve ODE for deterministic SIR
solution = odeint(SIR_model, [S0, I0, R0], t, args=(beta, gamma))
S_det, I_det, R_det = solution.T

# Run Gillespie algorithm multiple times
num_runs = 1000
all_S = []
all_I = []
all_R = []

for _ in range(num_runs):
    times, S_values, I_values, R_values = gillespie_SIR_scaled(S0, I0, R0, beta, gamma, max_time)
    all_S.append(np.interp(t, times, S_values))
    all_I.append(np.interp(t, times, I_values))
    all_R.append(np.interp(t, times, R_values))

# Compute average and standard deviation
avg_S = np.mean(all_S, axis=0)
std_S = np.std(all_S, axis=0)
avg_I = np.mean(all_I, axis=0)
std_I = np.std(all_I, axis=0)
avg_R = np.mean(all_R, axis=0)
std_R = np.std(all_R, axis=0)

# Plotting
plt.figure(figsize=(10,6))

# Deterministic SIR
plt.plot(t, S_det, label="Susceptible (Deterministic)", color="blue", linestyle="--")
plt.plot(t, I_det, label="Infectious (Deterministic)", color="red", linestyle="--")
plt.plot(t, R_det, label="Recovered (Deterministic)", color="green", linestyle="--")

# Average Stochastic SIR
plt.plot(t, avg_S, label="Avg Susceptible (Stochastic)", color="blue")
plt.fill_between(t, avg_S - std_S, avg_S + std_S, color="blue", alpha=0.2)
plt.plot(t, avg_I, label="Avg Infectious (Stochastic)", color="red")
plt.fill_between(t, avg_I - std_I, avg_I + std_I, color="red", alpha=0.2)
plt.plot(t, avg_R, label="Avg Recovered (Stochastic)", color="green")
plt.fill_between(t, avg_R - std_R, avg_R + std_R, color="green", alpha=0.2)

plt.xlabel("Time")
plt.ylabel("Population")
plt.title(f"Comparison of Deterministic and Stochastic SIR Models ({num_runs} runs)")
plt.legend()
plt.grid(True)
plt.show()