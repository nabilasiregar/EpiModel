import numpy as np
import matplotlib.pyplot as plt

# Introduce time-varying beta
def beta_t(t, beta_0, A, f):
    return beta_0 * (1 + A * np.sin(2 * np.pi * f * t))

def GA_with_demography_and_forcing(S0, I0, R0, beta_0, A, f, gamma, Lambda, mu, max_time):
    # Initial conditions
    S, I, R = S0, I0, R0
    t = 0
    times = [t]
    S_values = [S]
    I_values = [I]
    R_values = [R]
       

    while t < max_time and I > 0:
            # Rest of the code is similar]
        current_beta = beta_t(t, beta_0, A, f)
        N = S + I + R

        # Calculate propensities with the current beta
        a1 = current_beta * S * I / N        # Transmission
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
        # ... [same event determination logic as before]

        # Store results
        times.append(t)
        S_values.append(S)
        I_values.append(I)
        R_values.append(R)

    return times, S_values, I_values, R_values

# Example use:

S0, I0, R0 = 990, 10, 0
beta_0 = 0.3
A = 0.05  # amplitude of periodic forcing
f = 0.01  # frequency of periodic forcing
gamma = 0.1
Lambda = 5
mu = 0.01
max_time = 200

times, S_values, I_values, R_values = GA_with_demography_and_forcing(S0, I0, R0, beta_0, A, f, gamma, Lambda, mu, max_time)

# Plot
plt.figure(figsize=(10,6))
plt.plot(times, I_values, label="Infected (I)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Infections over Time with Periodic Forcing")
plt.legend()
plt.grid(True)
plt.show()
