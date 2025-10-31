# LIBRARIES
from models import pyworld2
from models.pyworld2.utils import plot_world_variables, plt
from report_helpers.plotting import generate_learning_plots
from report_helpers.timing import generate_timing_table
import itertools
import json
import math
import random
import numpy as np
import os
import pandas as pd
import time

# JSON UTILITY FUNCTIONS
def load_json_config(file_path):
    """
    Load configuration data from JSON file.
    
    Parameters:
    file_path: Path to the JSON configuration file
    
    Returns:
    Dictionary containing the loaded JSON data
    """
    with open(file_path, "r") as file:
        return json.load(file)

def save_json_config(data, file_path):
    """
    Save configuration data to JSON file.
    
    Parameters:
    data: Dictionary containing data to save
    file_path: Path where to save the JSON file
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def update_world2_parameters(data, birth_rate_normal, natural_resource_usage_normal, 
                           food_coefficient, capital_investment_discard_normal, pollution_normal):
    """
    Update World2 model parameters in the configuration data.
    
    Parameters:
    data: Configuration data dictionary
    birth_rate_normal: BRN - Birth Rate Normal [fraction/year] Base run 0.028
    natural_resource_usage_normal: NRUN - Natural-Resource Usage Normal Base run 0.25
    food_coefficient: FC - Food Coefficient [] Base run 0.8
    capital_investment_discard_normal: CIGN - Capital-Investment Discard Normal [fraction/year] Base run 0.03
    pollution_normal: POLN - Pollution Normal [pollution units/person/year]
    
    Returns:
    Updated configuration data dictionary
    """
    for entry in data:
        if "BRN1" in entry:
            entry["BRN1"] = birth_rate_normal
        elif "NRUN1" in entry:
            entry["NRUN1"] = natural_resource_usage_normal
        elif "POLN" in entry:
            entry["POLN"] = pollution_normal
        elif "FC1" in entry:
            entry["FC1"] = food_coefficient
        elif "CIGN1" in entry:
            entry["CIGN1"] = capital_investment_discard_normal
    return data

def check_parameter_feasibility(parameters, parameter_bounds):
    """
    Check if parameters are within feasible bounds.
    
    Parameters:
    parameters: List of parameter values to check
    parameter_bounds: List of [min, max] bounds for each parameter
    
    Returns:
    Boolean indicating whether all parameters are within bounds
    """
    for param_idx, param_value in enumerate(parameters):
        if not (parameter_bounds[param_idx][0] <= param_value <= parameter_bounds[param_idx][1]):
            return False
    return True

# MAIN EXECUTION
if __name__ == "__main__":
    
    # Start timing
    start_time = time.time()
    
    # EXPERIMENTAL PARAMETERS
    #epsilon_levels = [0.1, 0.4, 0.6, 0.8]  # Exploration probability levels
    epsilon_levels = [0.4, 0.6]  # Exploration probability levels
    rho_factor_levels = [[-0.01, 0, 0.01], [-0.1, 0, 0.1]]  # Parameter adjustment factors
    num_repetitions = 2  # Number of experimental repetitions
    
    # JSON configuration file path
    config_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pyworld2", "functions_switch_default.json")
    
    # TREATMENT COMBINATIONS
    treatments = list(itertools.product(epsilon_levels, rho_factor_levels, range(1, num_repetitions+1)))
    random.shuffle(treatments)  # Randomize execution order
    
    experimental_results = []
    
    # ALGORITHM EXECUTION
    for epsilon, rho_factors, repetition in treatments:
        treatment_start_time = time.time()
        
        # Initial parameter values for World2 model
        # [BRN, NRUN, FC, CIGN, POLN]
        current_parameters = [0.028, 0.25, 0.8, 0.03, 0.5]
        
        # Parameter feasibility bounds
        parameter_bounds = [[0.02, 0.04], [0.1, 1.0], [0.6, 1.25], [0.02, 0.04], [0.1, 1.0]]
        
        # RL algorithm parameters
        num_actions = len(rho_factors)  # Number of possible actions per parameter
        q_table = np.zeros((num_actions, num_actions, num_actions, num_actions, num_actions))  # Q-table for 5 parameters
        max_runs = 50  # Maximum number of learning iterations
        
        # Load initial configuration and get baseline return
        json_config = load_json_config(config_file_path)
        updated_config = update_world2_parameters(json_config, *current_parameters)
        save_json_config(updated_config, os.path.join("models", "pyworld2", "current_world2_config.json"))
        
        # Initialize World2 model and get baseline performance
        world2_model = pyworld2.World2()
        world2_model.set_state_variables()
        world2_model.set_initial_state()
        world2_model.set_table_functions()
        world2_model.set_switch_functions(os.path.join("models", "pyworld2", "current_world2_config.json"))
        world2_model.run()
        
        initial_return = world2_model.aveg_ql()  # Average Quality of Life
        learning_trajectory = [initial_return]
        
        # RL learning loop
        for run in range(max_runs):
            previous_parameters = list(current_parameters)  # Store previous parameter values
            
            # Load configuration and execute model with previous parameters
            json_config = load_json_config(config_file_path)
            updated_config = update_world2_parameters(json_config, *current_parameters)
            save_json_config(updated_config, os.path.join("models", "pyworld2", "current_world2_config.json"))
            
            # Run World2 model with current parameters
            world2_model = pyworld2.World2()
            world2_model.set_state_variables()
            world2_model.set_initial_state()
            world2_model.set_table_functions()
            world2_model.set_switch_functions(os.path.join("models", "pyworld2", "current_world2_config.json"))
            world2_model.run()
            
            previous_return = world2_model.aveg_ql()  # Previous performance metric
            
            # E-greedy action selection
            if epsilon < random.random():
                # Exploitation: select action with highest Q-value
                max_q_value = np.amax(q_table)
                max_indices = np.where(q_table == max_q_value)
                action_i, action_j, action_k, action_l, action_m = [max_indices[dim][0] for dim in range(q_table.ndim)]
            else:
                # Exploration: select random action
                action_i, action_j, action_k, action_l, action_m = [random.randint(0, num_actions-1) for _ in range(q_table.ndim)]
            
            # Apply selected actions to parameters
            current_parameters[0] *= (1 + rho_factors[action_i])  # Birth Rate Normal
            current_parameters[1] *= (1 + rho_factors[action_j])  # Natural Resource Usage Normal
            current_parameters[2] *= (1 + rho_factors[action_k])  # Food Coefficient
            current_parameters[3] *= (1 + rho_factors[action_l])  # Capital Investment Discard Normal
            current_parameters[4] *= (1 + rho_factors[action_m])  # Pollution Normal
            
            # Check parameter feasibility
            if check_parameter_feasibility(current_parameters, parameter_bounds):
                # Execute World2 model with new parameters
                json_config = load_json_config(config_file_path)
                updated_config = update_world2_parameters(json_config, *current_parameters)
                save_json_config(updated_config, os.path.join("models", "pyworld2", "current_world2_config.json"))
                
                world2_model = pyworld2.World2()
                world2_model.set_state_variables()
                world2_model.set_initial_state()
                world2_model.set_table_functions()
                world2_model.set_switch_functions(os.path.join("models", "pyworld2", "current_world2_config.json"))
                world2_model.run()
                
                current_return = world2_model.aveg_ql()  # Current performance metric
                
                # Update Q-table with relative improvement reward
                reward = (current_return - previous_return) / previous_return
                q_table[action_i][action_j][action_k][action_l][action_m] += reward
                
                # Revert to previous parameters if no improvement
                if previous_return >= current_return:
                    current_parameters = previous_parameters
            else:
                # Penalize infeasible parameter combinations
                q_table[action_i][action_j][action_k][action_l][action_m] += -100
                current_parameters = previous_parameters
            
            # Store learning trajectory
            learning_trajectory.append(previous_return)
        
        # Calculate treatment execution time
        treatment_end_time = time.time()
        treatment_duration = (treatment_end_time - treatment_start_time) / 60  # Convert to minutes
        
        # Store results for each run within the treatment
        for run_idx, return_value in enumerate(learning_trajectory):
            experimental_results.append([
                epsilon,
                rho_factors[-1],  # Use the maximum rho factor as identifier
                repetition,
                run_idx,
                treatment_duration,
                return_value
            ])
    
    # RESULTS PROCESSING AND EXPORT
    results_df = pd.DataFrame(experimental_results,
                             columns=['Epsilon_Level', 'Rho_Level', 'Repetition', 'Run', 'Execution_Time', 'Return'])
    
    # Sort results by experimental factors
    results_df = results_df.sort_values(by=['Epsilon_Level', 'Rho_Level', 'Repetition', 'Run'])
    
    # Export results to CSV
    output_filename = 'World2_Results.csv'
    results_df.to_csv(output_filename, index=False)
    
    print("World2 RL Experimental Results Summary:")
    print(results_df.head(10))
    print(f"\nTotal treatments executed: {len(treatments)}")
    print(f"Results exported to: {output_filename}")
    
    # Calculate and display total execution time
    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    print(f"Total execution time: {total_duration:.3f} minutes")
    
    # Generate learning plots
    print("\nGenerating learning curve plots...")
    generate_learning_plots(results_df, title="RL Agent Learning (World2)")
    
    # Generate timing statistics table
    print("Generating execution time statistics...")
    timing_stats = generate_timing_table(output_filename, max_runs)
    print("Timing statistics exported to: execution_time_statistics.csv")
    print(timing_stats)