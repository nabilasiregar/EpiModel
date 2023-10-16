import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I / (S + I + R)
    dI_dt = beta * S * I / (S + I + R) - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]

def gillespie_SIR(S0, I0, R0, beta, gamma, max_time):
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
        a1 = beta * S * I / N
        a2 = gamma * I
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
max_time = 160
t = np.linspace(0, max_time, 1000)

# Parameters for the experiment
R0_values = [0.5, 1, 1.5, 2, 2.5, 3]
N_values = [100, 500, 1000, 5000]
gamma = 0.1  # Fixed recovery rate for simplicity
num_runs = 100

results = []

for N in N_values:
    for R0 in R0_values:
        beta = R0 * gamma / N  # Calculate beta based on R0 and N
        extinction_times = []
        
        for _ in range(num_runs):
            times, _, I_values, _ = gillespie_SIR(N-1, 1, 0, beta, gamma, max_time)
            
            # Find the time of extinction (when I becomes 0)
            extinction_time = times[next((i for i, val in enumerate(I_values) if val == 0), len(times)-1)]
            extinction_times.append(extinction_time)
        
        # Calculate average time to extinction
        avg_extinction_time = np.mean(extinction_times)
        
        results.append({
            'N': N,
            'R0': R0,
            'avg_extinction_time': avg_extinction_time
        })

# Plotting
plt.figure(figsize=(10, 6))
for N in N_values:
    times = [res['avg_extinction_time'] for res in results if res['N'] == N]
    plt.plot(R0_values, times, label=f'N={N}', marker='o')

plt.title("Average Time to Extinction vs. R0 for Different Population Sizes")
plt.xlabel("R0")
plt.ylabel("Average Time to Extinction")
plt.legend()
plt.grid(True)
plt.show()
