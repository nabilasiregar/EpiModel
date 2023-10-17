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
S0 = 200
I0 = 5
R0 = 0
beta = 0.3
gamma = 0.1
max_time = 200
t = np.linspace(0, max_time, 1000)

solution = odeint(SIR_model, [S0, I0, R0], t, args=(beta, gamma))
S_det, I_det, R_det = solution.T

times, S_values, I_values, R_values = gillespie_SIR(S0, I0, R0, beta, gamma, max_time)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(t, S_det, label="Susceptible (Deterministic)", color="blue", linestyle="--")
plt.plot(t, I_det, label="Infectious (Deterministic)", color="red", linestyle="--")
plt.plot(t, R_det, label="Recovered (Deterministic)", color="green", linestyle="--")
plt.plot(times, S_values, label="Susceptible", color="blue")
plt.plot(times, I_values, label="Infected", color="red")
plt.plot(times, R_values, label="Recovered", color="green")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Stochastic SIR Model using Gillespie Algorithm")
plt.legend()
plt.grid(True)
plt.show()
