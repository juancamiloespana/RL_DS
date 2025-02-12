# Import necessary libraries
from LotkaVolterra_JSJ import LotkaVolterraModel
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
    lotka_volterra_instance = LotkaVolterraModel(*params)
    initial_reward, *_ = lotka_volterra_instance.simulate()

    print(f"Initial Parameters: {params}")
    print(f"Initial Reward: {round(initial_reward, 5)}")

    rewards_log = [initial_reward]

    # Optimization loop
    for i in tqdm(range(runs), desc="Optimizing Parameters"):
        #st=time.time()
        # Compute adaptive epsilon and step size
        epsilon = get_adaptive_epsilon(epsilon_0, epsilon_min, epsilon_decay, i)
        step_size = get_adaptive_step_size(step_0, step_min, step_decay, i)
        step_factors = [-step_size, 0, step_size]

        # Store previous parameters and reward
        prev_params = list(params)
        lotka_volterra_instance = LotkaVolterraModel(*prev_params)
        prev_reward, *_ = lotka_volterra_instance.simulate()
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
            current_reward, *_ = lotka_volterra_instance.simulate()

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

    return params, rewards_log

# Define initial parameters and bounds
initial_params = [0.002, 0.04, 0.1, 0.0025]
param_bounds = [[0.0015, 0.0025], [0.03, 0.05], [0.05, 0.15], [0.002, 0.003]]

# Run the optimization
optimized_params = optimize_lotka_volterra(initial_params, param_bounds)


import time
st=time.time()
lotka_volterra_instance = LotkaVolterraModel(*optimized_params)
current_reward, *_ = lotka_volterra_instance.simulate()
et=time.time()
print(f"Final Reward: {round(current_reward, 5)}")
print(f"Time taken: {et-st:.2f} seconds")


P_3000_runs=[0.0023824491246225493,0.030653060921676308, 0.1485923302075364, 0.002000433195741189]
lotka_volterra_instance = LotkaVolterraModel(*P_4000_runs)
current_reward, *_ = lotka_volterra_instance.simulate()

P_4000_runs=[0.0024798504881202326,0.030055751663851004,0.14993275482788457,0.0020000123410408217]

#### Original version ###############################################

# Import necessary libraries
from LotkaVolterra_JSJ import LotkaVolterraModel
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to check if parameters remain within feasible limits
def is_within_bounds(params, bounds):
    """
    Verifies if the given parameter values remain within defined feasibility limits.

    :param params: List of parameter values
    :param bounds: List of tuples (min, max) for each parameter
    :return: True if all parameters are within bounds, else False
    """
    return all(bounds[i][0] <= params[i] <= bounds[i][1] for i in range(len(params)))

# Function to optimize Lotka-Volterra parameters using the ε-greedy strategy
def optimize_lotka_volterra(
    initial_params, param_bounds, exploration_rate=0.1, step_percentage=0.01, runs=1000
):
    """
    Optimizes the Lotka-Volterra model parameters using a K-Armed Bandit algorithm.

    :param initial_params: Initial parameter values
    :param param_bounds: Bounds for each parameter
    :param exploration_rate: Probability of exploration (ε)
    :param step_percentage: Percentage change for parameter adjustments
    :param runs: Number of iterations for optimization
    :return: Optimized parameters, Final reward
    """

    # Define step size modifications
    step_factors = [-step_percentage, 0, step_percentage]
    num_actions = len(step_factors)  # Number of possible actions per parameter

    # Initialize Q-table for storing rewards
    q_table = np.zeros((num_actions, num_actions, num_actions, num_actions))
    num_dimensions = q_table.ndim  # Dimensionality of Q-table

    # Initialize parameters
    parameters = list(initial_params)
    lotka_volterra_instance = LotkaVolterraModel(*parameters)
    initial_reward, *_ = lotka_volterra_instance.simulate()

    print(f"Initial Parameters: {parameters}")
    print(f"Initial Reward: {round(initial_reward, 5)}")

    # Store reward values for plotting
    reward_log = [initial_reward]

    # Optimization loop
    for iteration in tqdm(range(runs), desc="Optimizing Parameters"):

        # Store previous parameters and reward
        previous_parameters = list(parameters)
        lotka_volterra_instance = LotkaVolterraModel(*previous_parameters)
        previous_reward, *_ = lotka_volterra_instance.simulate()
        reward_log.append(previous_reward)

        # Choose action based on exploration or exploitation
        if random.random() < exploration_rate:
            # Exploration: Select random action
            action_indices = [random.randint(0, num_actions - 1) for _ in range(num_dimensions)]
        else:
            # Exploitation: Select best action from Q-table
            best_q_value = np.amax(q_table)  # Identify maximum Q-value
            action_indices = [np.where(q_table == best_q_value)[i][0] for i in range(num_dimensions)]

        # Apply selected actions to update parameters
        parameters = [
            parameters[i] * (1 + step_factors[action_indices[i]])
            for i in range(len(parameters))
        ]

        # Check if new parameters are within feasibility limits
        if is_within_bounds(parameters, param_bounds):
            # Compute reward for new parameter set
            lotka_volterra_instance = LotkaVolterraModel(*parameters)
            current_reward, *_ = lotka_volterra_instance.simulate()

            # Update Q-table based on reward improvement
            q_table[tuple(action_indices)] += (current_reward - previous_reward) / previous_reward

            # Retain better parameter set; otherwise, revert
            if previous_reward >= current_reward:
                parameters = previous_parameters
        else:
            # Penalize infeasible solutions and revert parameters
            q_table[tuple(action_indices)] -= 100
            parameters = previous_parameters

    # Final results
    lotka_volterra_instance = LotkaVolterraModel(*parameters)
    final_reward, *_ = lotka_volterra_instance.simulate()
    
    print(f"Final Optimized Parameters: {parameters}")
    print(f"Final Reward: {round(final_reward, 5)}")

    # Generate optimization performance plot
    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.title("Optimization of Lotka-Volterra Model Using K-Armed Bandit Algorithm")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function Value")
    plt.text(0.1 * runs, max(reward_log) * 0.9, 
             f"Initial: {round(reward_log[0], 2)}, Final: {round(reward_log[-1], 2)}")
    plt.text(0.1 * runs, max(reward_log) * 0.85, 
             f"Improvement: {round((reward_log[-1] - reward_log[0]) / reward_log[0] * 100, 2)}%")
    plt.plot(range(runs + 1), reward_log, label="Reward")
    plt.legend()
    plt.show()

    return parameters, final_reward

# Define initial parameters and bounds
initial_parameters = [0.002, 0.04, 0.1, 0.0025]
parameter_bounds = [[0.0015, 0.0025], [0.03, 0.05], [0.05, 0.15], [0.002, 0.003]]

# Run the optimization function
optimized_params, final_reward = optimize_lotka_volterra(
    initial_parameters, parameter_bounds, exploration_rate=0.1, step_percentage=0.01, runs=4000
)



initial_parameters = [0.002, 0.04, 0.1, 0.0025]
parameter_bounds = [[0.0015, 0.0025], [0.03, 0.05], [0.05, 0.15], [0.002, 0.003]]





from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


# Define Bayesian Optimization Search Space
search_space = [
    Real(0.001, 0.9, name="exploration_rate"),  # Search ε between 0.05 and 0.5
    Real(0.001, 0.50, name="step_percentage")  # Search step size between 0.005 and 0.05
]

# Define objective function for Bayesian Optimization
@use_named_args(search_space)
def objective(exploration_rate, step_percentage):
    return optimize_lotka_volterra(initial_parameters, parameter_bounds, exploration_rate, step_percentage, runs=1000)

# Run Bayesian Optimization
result = gp_minimize(objective, search_space, n_calls=20, random_state=42, acq_func="EI")

# Extract best parameters
best_hyperparams = dict(zip(["exploration_rate", "step_percentage"], result.x))
best_reward = -result.fun  # Convert back from negative reward

# Print best parameters
print("\nBest Tuned Parameters:")
for key, val in best_hyperparams.items():
    print(f"{key}: {val:.6f}")
print(f"\nMaximum Achieved Reward: {best_reward:.6f}")

# Plot Convergence
plt.figure(figsize=(8, 5))
plt.plot(result.func_vals, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Negative Best Reward (Higher is Better)")
plt.title("Bayesian Optimization Convergence")
plt.grid()
plt.show()

# Run the optimization function with the best-found parameters
optimized_params, final_reward = optimize_lotka_volterra(
    initial_parameters, parameter_bounds,
    exploration_rate=best_hyperparams["exploration_rate"],
    step_percentage=best_hyperparams["step_percentage"],
    runs=5000  # Full optimization with the best parameters
)

print("\nFinal Optimized Parameters:", optimized_params)
print("Final Achieved Reward:", round(final_reward, 6))