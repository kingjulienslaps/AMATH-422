import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import curve_fit

# Simulating state transitions (from the previous code)
rng = np.random.default_rng()

A = np.array([[0.98, 0.10, 0],
              [0.02, 0.70, 0.05],
              [0, 0.20, 0.95]])

Tmax = 1000
states = np.zeros(Tmax, dtype=int)
states[0] = 0 #start in closed states

for t in np.arange(Tmax - 1):
    r = rng.uniform(0, 1)
    
    current_state = states[t]
    
    if current_state == 0:
        if r < A[0, 0]:
            states[t + 1] = 0
        else:
             states[t + 1] = 1
            
    elif current_state == 1:
        if r < A[1, 1]:
            states[t + 1] = 1
        elif A[1, 1] < r < A[1, 1] + A[1, 0]:
            states[t + 1] = 1
        else:
            states[t + 1] = 2
    
    elif current_state == 2:
        if r < A[2, 2]:
            states[t + 1] = 2
        else:
            states[t + 1] = 1

# Step 1: Identify closed states (0 and 1)
closed_states = (states == 0) | (states == 1)

# Step 2: Compute dwell times in the closed state
dwell_times = []
current_dwell = 0
for i in range(Tmax):
    if closed_states[i]:
        current_dwell += 1
    else:
        if current_dwell > 0:
            dwell_times.append(current_dwell)
            current_dwell = 0
# Add any ongoing dwell at the end
if current_dwell > 0:
    dwell_times.append(current_dwell)

# Step 3: Make a histogram of dwell times
plt.hist(dwell_times, bins=20, density=True, alpha=0.6, color='g')
plt.xlabel('Dwell time in closed state')
plt.ylabel('Frequency')
plt.title('Histogram of Dwell Times in Closed State')
plt.show()

# Step 4: Fit dwell times to a single exponential

# Define exponential distribution function
def exp_func(x, a, b):
    return a * np.exp(-b * x)

# Generate histogram data for fitting
hist_data, bin_edges = np.histogram(dwell_times, bins=20, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Perform the curve fit
popt, pcov = curve_fit(exp_func, bin_centers, hist_data)

# Step 5: Plot histogram and fitted exponential curve
plt.hist(dwell_times, bins=20, density=True, alpha=0.6, color='g', label='Histogram of dwell times')
plt.plot(bin_centers, exp_func(bin_centers, *popt), 'r--', label=f'Exponential fit: a={popt[0]:.2f}, b={popt[1]:.2f}')
plt.xlabel('Dwell time in closed state')
plt.ylabel('Frequency')
plt.title('Fit of Dwell Times to Exponential Distribution')
plt.legend()
plt.show()
