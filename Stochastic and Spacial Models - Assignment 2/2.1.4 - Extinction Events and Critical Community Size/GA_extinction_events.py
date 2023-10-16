import numpy as np
import matplotlib.pyplot as plt

# Gillespie SIR function
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

# Parameters for the experiment
R0_values = np.concatenate([
    np.arange(0.1, 1, 0.1),  # Fine intervals from 0.1 to 1
    np.arange(1, 10, 0.5),  # Medium intervals from 1 to 10
    np.arange(10, 51, 5)    # Coarse intervals from 10 to 50
])
gamma = 0.1  # Fixed recovery rate for simplicity
N = 1000  # Fixed population size for simplicity
num_runs = 100
max_time = 500  # Maximum simulation time

results = []

for R0 in R0_values:
    beta = R0 * gamma  # Calculate beta based on R0
    extinction_times = []
    
    for _ in range(num_runs):
        times, _, I_values, _ = gillespie_SIR(N-1, 1, 0, beta, gamma, max_time)
        
        # Find the time of extinction (when I becomes 0)
        extinction_time = times[next((i for i, val in enumerate(I_values) if val == 0), len(times)-1)]
        extinction_times.append(extinction_time)
    
    # Calculate average time to extinction
    avg_extinction_time = np.mean(extinction_times)
    
    results.append({
        'R0': R0,
        'avg_extinction_time': avg_extinction_time
    })

# Plotting
plt.figure(figsize=(10, 6))
times = [res['avg_extinction_time'] for res in results]
plt.plot(R0_values, times, marker='o', linestyle='-')

plt.title("Average Time to Extinction vs. R0")
plt.xlabel("R0")
plt.ylabel("Average Time to Extinction")
plt.grid(True)
plt.show()
