#### Original version ###############################################

# Import necessary libraries
from LotkaVolterra import LotkaVolterraModel
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