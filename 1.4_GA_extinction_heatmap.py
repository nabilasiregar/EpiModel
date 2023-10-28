import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams['text.usetex'] = True

def gillespie_SIR(S0, I0, R0, beta, gamma, max_time):
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

R0_values = np.concatenate([
    np.arange(0.1, 1, 0.1),
    np.arange(1, 10, 0.5)
])
N_values = [10, 50, 100, 500, 1000]
gamma = 0.1  
num_runs = 1000
max_time = 500  

results = np.zeros((len(N_values), len(R0_values)))

for i, N in enumerate(N_values):
    for j, R0 in enumerate(R0_values):
        beta = R0 * gamma  
        extinction_times = []
        
        for _ in range(num_runs):
            times, _, I_values, _ = gillespie_SIR(N-1, 1, 0, beta, gamma, max_time)
            extinction_time = times[next((i for i, val in enumerate(I_values) if val == 0), len(times)-1)]
            extinction_times.append(extinction_time)
        
        # Calculate average time to extinction
        avg_extinction_time = np.mean(extinction_times)
        
        results[i, j] = avg_extinction_time

# Plotting
plt.figure(figsize=(12, 7))
c = plt.contourf(R0_values, N_values, results, cmap='viridis', levels=100)
plt.colorbar(c, label="Average Time to Extinction (Timesteps)")
# plt.title(r"Impact of $R_0$ and $N$ on Time to Extinction")
plt.xlabel("$R_0$")
plt.ylabel("Population Size (N)")
plt.grid(True)
plt.show()
