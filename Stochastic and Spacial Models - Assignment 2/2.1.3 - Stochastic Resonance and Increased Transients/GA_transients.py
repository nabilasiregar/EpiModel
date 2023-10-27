import numpy as np
import matplotlib.pyplot as plt
import GA_stochastic_res_V2 as GA


N_values = np.linspace(100, 5000, 25).astype(int)  
R0_values = np.linspace(2.0, 10.0, 25)


transient_S_matrix = np.zeros((len(N_values), len(R0_values)))
transient_I_matrix = np.zeros((len(N_values), len(R0_values)))

for i, N in enumerate(N_values):
    for j, R0 in enumerate(R0_values):
        beta = R0 * GA.gamma
        
        I0 = 5
        S0 = N - I0 - R0

        S_accum = np.zeros_like(GA.t_common)
        I_accum = np.zeros_like(GA.t_common)
        
        for _ in range(GA.num_runs):
            times, S_stoch, I_stoch, R_stoch = GA.GA_with_demography(S0, I0, R0, beta, GA.gamma, GA.Lambda, GA.mu, GA.max_time)
            
            S_stoch_interp = np.interp(GA.t_common, times, S_stoch)
            I_stoch_interp = np.interp(GA.t_common, times, I_stoch)
            
            S_accum += S_stoch_interp
            I_accum += I_stoch_interp
        
        S_stoch_mean = S_accum / GA.num_runs
        I_stoch_mean = I_accum / GA.num_runs
        
        solution = GA.odeint(GA.SIR_demography, [S0, I0, R0], GA.t, args=(beta, GA.gamma, GA.Lambda, GA.mu))
        S_det, I_det, R_det = solution.T
        
        transient_S = GA.measure_transient(S_stoch_mean, S_det)
        transient_I = GA.measure_transient(I_stoch_mean, I_det)
        
        transient_S_matrix[i, j] = transient_S
        transient_I_matrix[i, j] = transient_I

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

c1 = ax[0].imshow(transient_S_matrix, cmap='viridis', aspect='auto', origin='lower',
                  extent=[min(R0_values), max(R0_values), min(N_values), max(N_values)])
fig.colorbar(c1, ax=ax[0])
ax[0].set_title('Relative Transient for Susceptible')
ax[0].set_xlabel('R0')
ax[0].set_ylabel('N')

c2 = ax[1].imshow(transient_I_matrix, cmap='viridis', aspect='auto', origin='lower',
                  extent=[min(R0_values), max(R0_values), min(N_values), max(N_values)])
fig.colorbar(c2, ax=ax[1])
ax[1].set_title('Relative Transient for Infectious')
ax[1].set_xlabel('R0')
ax[1].set_ylabel('N')

plt.tight_layout()
plt.show()
