import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams['text.usetex'] = True
from scipy.optimize import curve_fit


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

N_values = np.concatenate([
    np.arange(100, 1000, 100),      # Medium intervals from 100 to 1000
    np.arange(1000, 10000, 100),  # Coarse intervals from 1000 to 10000
])

R0 = 3 # Let's use R0 = 2 as an example
gamma = 0.1  # Fixed recovery rate for simplicity
beta = R0 * gamma  # Calculate beta based on fixed R0
num_runs = 100
max_time = 500  # Maximum simulation time

results = []

for N in N_values:
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
        'avg_extinction_time': avg_extinction_time
    })

# Extract 'avg_extinction_time' from results and assign to times
times = [res['avg_extinction_time'] for res in results]

# Exponential decay function
def exp_decay(x, a, b, c):
    return a * np.log(b * x) + c

# Fit the curve
popt, pcov = curve_fit(exp_decay, N_values, times, p0=(1, 0.1, 1))
a, b, c = popt

# Print the fitted parameters
perr = np.sqrt(np.diag(pcov))
print(f"a = {popt[0]:.5f} ± {perr[0]:.5f}")
print(f"b = {popt[1]:.5f} ± {perr[1]:.5f}")
print(f"c (convergence value) = {popt[2]:.5f} ± {perr[2]:.5f}")


# Plotting original data and the fit
plt.figure(figsize=(10, 6))
plt.plot(N_values, times, '-o', color='darkcyan', label='Individual extinction time', markersize=6)
plt.plot(N_values, exp_decay(N_values, *popt), 'darkorange', linestyle="--", label='Extinction convergence curve')
plt.xlabel("Population Size (N)")
plt.ylabel("Average Time to Extinction (Timesteps)")
plt.legend()
plt.grid(True)
plt.show()
