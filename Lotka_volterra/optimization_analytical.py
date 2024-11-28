import numpy as np
from scipy.optimize import minimize

simulation_count = 0
# Function to simulate the predator-prey dynamics
def simulate(params, R0, W0, T, dt):
    global simulation_count
    simulation_count += 1  # 
    alpha, beta, delta, gamma = params
    rabbits = R0
    wolves = W0
    time_steps = int(T / dt)
    wolf_population = []  # To calculate the average later
    
    for _ in range(time_steps):
        # Calculate birth and death rates
        wolves_birth = alpha * rabbits * wolves
        wolves_death = beta * wolves
        rabbits_birth = delta * rabbits
        rabbits_death = gamma * rabbits * wolves
        
        # Update populations
        wolves += (wolves_birth - wolves_death) * dt
        rabbits += (rabbits_birth - rabbits_death) * dt
        
        # Store wolf population
        wolf_population.append(wolves)
    
    # Return the long-run average wolf population
    return np.mean(wolf_population)

# Objective function to maximize wolf population
def objective(params, R0, W0, T, dt):
    return -simulate(params, R0, W0, T, dt)  # Negative because we maximize

# Initial conditions and parameter bounds
R0, W0 = 100, 25  # Initial populations
P = [0.002, 0.04, 0.1, 0.0025]  # Initial parameter values [alpha, beta, delta, gamma]
Lim = [[0.0015, 0.0025], [0.03, 0.05], [0.05, 0.15], [0.002, 0.003]]  # Parameter bounds

# Optimization
T = 40000  # Simulation time
dt = 0.01  # Time step
result = minimize(objective, P, args=(R0, W0, T, dt), bounds=Lim)
optimal_params = result.x

# Output results
print("Optimal Parameters:")
print(f"Alpha (predator birth): {optimal_params[0]:.5f}")
print(f"Beta (predator death): {optimal_params[1]:.5f}")
print(f"Delta (prey birth): {optimal_params[2]:.5f}")
print(f"Gamma (prey death): {optimal_params[3]:.5f}")
print("\nMaximum Average Wolf Population:")
print(f"{-result.fun:.5f}")

optimal_params
simulation_count

result