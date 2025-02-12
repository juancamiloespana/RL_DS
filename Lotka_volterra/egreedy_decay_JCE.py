# Import necessary libraries
from LotkaVolterra import LotkaVolterraModel
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to check if parameters are within bounds
def is_within_bounds(params, bounds):
    return all(bounds[i][0] <= params[i] <= bounds[i][1] for i in range(len(params)))

# Function to compute decay rate dynamically
def compute_decay_rate(initial_value, final_value, total_iterations):
    """
    Computes the exponential decay rate required to smoothly transition 
    from an initial value to a final value over a specified number of iterations.

    :param initial_value: The starting value
    :param final_value: The ending value
    :param total_iterations: The total number of iterations for decay
    :return: Computed decay rate
    """
    return -np.log(final_value / initial_value) / total_iterations

# Function to update exploration rate (ε) adaptively
def get_adaptive_epsilon(initial_epsilon, epsilon_min, decay_rate, iteration):
    return max(epsilon_min, initial_epsilon * np.exp(-decay_rate * iteration))

# Function to update step size adaptively
def get_adaptive_step_size(initial_step, step_min, decay_rate, iteration):
    return max(step_min, initial_step * np.exp(-decay_rate * iteration))

# Function to update parameters based on selected actions
def update_parameters(params, step_factors, actions):
    return [params[i] * (1 + step_factors[actions[i]]) for i in range(len(params))]

# Function to optimize Lotka-Volterra parameters using epsilon-greedy method
def optimize_lotka_volterra(
    initial_params, param_bounds, runs=4000, epsilon_0=0.8, epsilon_min=0.01, step_0=0.20, step_min=0.0001
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
    #print(type(initial_params))
    lotka_volterra_instance = LotkaVolterraModel(*params)
    initial_reward = lotka_volterra_instance.simulate()

    #print(f"Initial Parameters: {params}")
    #print(f"Initial Reward: {round(initial_reward, 5)}")

    rewards_log = [initial_reward]

    # Optimization loop
    for i in range(runs):
        #st=time.time()
        # Compute adaptive epsilon and step size
        epsilon = get_adaptive_epsilon(epsilon_0, epsilon_min, epsilon_decay, i)
        step_size = get_adaptive_step_size(step_0, step_min, step_decay, i)
        step_factors = [-step_size, 0, step_size]

        # Store previous parameters and reward
        prev_params = list(params)
        lotka_volterra_instance = LotkaVolterraModel(*prev_params)
        prev_reward = lotka_volterra_instance.simulate()
        rewards_log.append(prev_reward)

        # Choose actions based on exploration or exploitation
        if epsilon < random.random():
            # Exploitation: Use best-known actions
            best_q_value = np.amax(q_table)
            best_action_indices = np.where(q_table == best_q_value)
            actions = [best_action_indices[i][0] for i in range(q_table.ndim)]
        else:
            # Exploration: Choose random actions
            actions = [random.randint(0, num_actions - 1) for _ in range(q_table.ndim)]

        # Update parameters
        params = update_parameters(prev_params, step_factors, actions)

        # Check if new parameters are within bounds
        if is_within_bounds(params, param_bounds):
            # Evaluate the new parameter set
            lotka_volterra_instance = LotkaVolterraModel(*params)
            current_reward= lotka_volterra_instance.simulate()

            # Update Q-table with the reward difference
            q_table[tuple(actions)] += ((current_reward - prev_reward) / prev_reward)

            # If the new reward is worse, revert to previous parameters
            if prev_reward >= current_reward:
                params = prev_params
        else:
            # Penalize infeasible solutions and revert parameters
            q_table[tuple(actions)] += -100
            params = prev_params
        #et=time.time()
        #print(f"Time taken: {et-st:.2f} seconds")

    # print(f"Final Optimized Parameters: {params}")

    # # Plot reward progression
    # plt.figure(figsize=(8, 5))
    # plt.plot(rewards_log, label="Reward over iterations")
    # plt.xlabel("Iteration")
    # plt.ylabel("Reward")
    # plt.title("Optimization Progress")
    # plt.legend()
    # plt.grid()
    # plt.show()


    return params, rewards_log

