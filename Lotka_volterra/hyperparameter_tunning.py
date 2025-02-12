# Import necessary libraries
from LotkaVolterra_JSJ import LotkaVolterraModel
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Function to check if parameters are within bounds
def is_within_bounds(params, bounds):
    return all(bounds[i][0] <= params[i] <= bounds[i][1] for i in range(len(params)))

# Function to compute decay rate dynamically
def compute_decay_rate(initial_value, final_value, total_iterations):
    return -np.log(final_value / initial_value) / total_iterations

# Function to update exploration rate (ε) adaptively
def get_adaptive_epsilon(initial_epsilon, epsilon_min, decay_rate, iteration):
    return max(epsilon_min, initial_epsilon * np.exp(-decay_rate * iteration))

# Function to update step size adaptively
def get_adaptive_step_size(initial_step, step_min, decay_rate, iteration):
    return max(step_min, initial_step * np.exp(-decay_rate * iteration))

# Function to sample random parameters within given bounds
def sample_random_parameters(param_bounds):
    return [random.uniform(bounds[0], bounds[1]) for bounds in param_bounds]

# Function to update parameters based on selected actions
def update_parameters(params, step_factors, actions):
    return [params[i] * (1 + step_factors[actions[i]]) for i in range(len(params))]

# Function to optimize Lotka-Volterra parameters using epsilon-greedy method
def optimize_lotka_volterra(
    initial_params, param_bounds, runs=3000, epsilon_0=0.8, epsilon_min=0.01, step_0=0.50, step_min=0.001
):
    """
    Optimizes the Lotka-Volterra model parameters using the epsilon-greedy strategy.

    :param initial_params: Initial parameter values
    :param param_bounds: Bounds for each parameter
    :param runs: Number of iterations for optimization
    :param epsilon_0: Initial exploration rate
    :param epsilon_min: Minimum exploration rate
    :param step_0: Initial step size
    :param step_min: Minimum step size
    :return: Final optimized parameters
    """
    
    # Automatically compute decay rates based on given initial and final values
    epsilon_decay = compute_decay_rate(epsilon_0, epsilon_min, runs)
    step_decay = compute_decay_rate(step_0, step_min, runs)

    # Initialize Q-table for storing rewards
    num_actions = 3  # Number of possible actions per parameter
    q_table = np.zeros((num_actions,) * len(initial_params))

    # Initialize parameters
    params = list(initial_params)
    lotka_volterra_instance = LotkaVolterraModel(*params)
    initial_reward, *_ = lotka_volterra_instance.simulate()

    print(f"Initial Parameters: {params}")
    print(f"Initial Reward: {round(initial_reward, 5)}")

    rewards_log = [initial_reward]

    # Optimization loop
    for i in tqdm(range(runs), desc="Optimizing Parameters"):
        #st = time.time()
        # Compute adaptive epsilon and step size
        epsilon = get_adaptive_epsilon(epsilon_0, epsilon_min, epsilon_decay, i)
        step_size = get_adaptive_step_size(step_0, step_min, step_decay, i)
        step_factors = [-step_size, 0, step_size]

        # Store previous parameters and reward
        prev_params = list(params)
        lotka_volterra_instance = LotkaVolterraModel(*prev_params)
        prev_reward, *_ = lotka_volterra_instance.simulate()
        rewards_log.append(prev_reward)

        if epsilon < random.random():
            # Exploitation: Use best-known actions from Q-table
            best_q_value = np.amax(q_table)
            best_action_indices = np.where(q_table == best_q_value)
            actions = [best_action_indices[i][0] for i in range(q_table.ndim)]
            params = update_parameters(prev_params, step_factors, actions)
        else:
            # Exploration: Select completely random parameter values
            params = sample_random_parameters(param_bounds)
            actions = None  # No Q-table update for exploration

        # Check if new parameters are within bounds
        if is_within_bounds(params, param_bounds):
            # Evaluate the new parameter set
            lotka_volterra_instance = LotkaVolterraModel(*params)
            current_reward, *_ = lotka_volterra_instance.simulate()

            # Update Q-table only if exploitation was performed
            if actions is not None:
                q_table[tuple(actions)] += ((current_reward - prev_reward) / prev_reward)

            # If the new reward is worse, revert to previous parameters
            if prev_reward >= current_reward:
                params = prev_params
        else:
            # Penalize infeasible solutions only if exploitation was used
            if actions is not None:
                q_table[tuple(actions)] += -100
            params = prev_params

        #et = time.time()
        #print(f"Time taken: {et - st:.2f} seconds")

    print(f"Final Optimized Parameters: {params}")

    # Plot reward progression
    plt.figure(figsize=(8, 5))
    plt.plot(rewards_log, label="Reward over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Optimization Progress")
    plt.legend()
    plt.grid()
    plt.show()

    return params

# Define initial parameters and bounds
initial_params = [0.002, 0.04, 0.1, 0.0025]
param_bounds = [[0.0015, 0.0025], [0.03, 0.05], [0.05, 0.15], [0.002, 0.003]]

# Run the optimization
optimized_params = optimize_lotka_volterra(initial_params, param_bounds)

# Evaluate final optimized parameters
st = time.time()
lotka_volterra_instance = LotkaVolterraModel(*optimized_params)
current_reward, *_ = lotka_volterra_instance.simulate()
et = time.time()

print(f"Final Reward: {round(current_reward, 5)}")
print(f"Time taken: {et - st:.2f} seconds")
