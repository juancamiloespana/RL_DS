# LIBRARIES
from models.LotkaVolterra import LotkaVolterraModel
from report_helpers.plotting import generate_learning_plots
from report_helpers.timing import generate_timing_table
import numpy as np
import random
import pandas as pd
import itertools
import time

# FEASIBILITY CHECK FUNCTION
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
    epsilon_levels = [0.6, 0.8]  # Exploration probability levels
    rho_factor_levels = [[-0.001, 0, 0.001], [-0.01, 0, 0.01], [-0.1, 0, 0.1]]  # Parameter adjustment factors
    num_repetitions = 3# Number of experimental repetitions #Eran 50
    
    # TREATMENT COMBINATIONS
    treatments = list(itertools.product(epsilon_levels, rho_factor_levels, range(1, num_repetitions+1)))
    random.shuffle(treatments)  # Randomize execution order
    
    experimental_results = []
    
    # ALGORITHM EXECUTION
    for epsilon, rho_factors, repetition in treatments:
        treatment_start_time = time.time()
        
        # Initial parameter values for Lotka-Volterra model
        current_parameters = [0.002, 0.04, 0.1, 0.0025]
        
        # Parameter feasibility bounds
        parameter_bounds = [[0.0015, 0.0025], [0.03, 0.05], [0.05, 0.15], [0.002, 0.003]]
        
        # RL algorithm parameters
        num_actions = len(rho_factors)  # Number of possible actions per parameter
        q_table = np.zeros((num_actions, num_actions, num_actions, num_actions))  # Q-table for 4 parameters
        max_runs = 500  # Maximum number of learning iterations
        
        # Initialize Lotka-Volterra model and get baseline return
        lv_model = LotkaVolterraModel(*current_parameters)
        initial_return = lv_model.simulate()
        
        learning_trajectory = [initial_return]
        
        # RL learning loop
        for run in range(max_runs):
            previous_parameters = list(current_parameters)  # Store previous parameter values
            lv_model_prev = LotkaVolterraModel(*previous_parameters)
            previous_return = lv_model_prev.simulate()
            
            # E-greedy action selection
            if epsilon < random.random():
                # Exploitation: select action with highest Q-value
                max_q_value = np.amax(q_table)
                max_indices = np.where(q_table == max_q_value)
                action_i, action_j, action_k, action_l = [max_indices[dim][0] for dim in range(q_table.ndim)]
            else:
                # Exploration: select random action
                action_i, action_j, action_k, action_l = [random.randint(0, num_actions-1) for _ in range(q_table.ndim)]
            
            # Apply selected actions to parameters
            current_parameters[0] = current_parameters[0] * (1 + rho_factors[action_i])
            current_parameters[1] = current_parameters[1] * (1 + rho_factors[action_j])
            current_parameters[2] = current_parameters[2] * (1 + rho_factors[action_k])
            current_parameters[3] = current_parameters[3] * (1 + rho_factors[action_l])
            
            # Check parameter feasibility
            if check_parameter_feasibility(current_parameters, parameter_bounds):
                # Calculate return for feasible parameters
                lv_model_current = LotkaVolterraModel(*current_parameters)
                current_return = lv_model_current.simulate()
                
                # Update Q-table with relative improvement reward
                reward = (current_return - previous_return) / previous_return
                q_table[action_i][action_j][action_k][action_l] += reward
                
                # Revert to previous parameters if no improvement
                if previous_return >= current_return:
                    current_parameters = previous_parameters
            else:
                # Penalize infeasible parameter combinations
                q_table[action_i][action_j][action_k][action_l] += -100
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
    output_filename = 'LotkaVolterra_Results.csv'
    results_df.to_csv(output_filename, index=False)
    
    print("Experimental Results Summary:")
    print(results_df.head(10))
    print(f"\nTotal treatments executed: {len(treatments)}")
    print(f"Results exported to: {output_filename}")
    
    # Calculate and display total execution time
    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    print(f"Total execution time: {total_duration:.3f} minutes")
    
    # Generate learning plots
    print("\nGenerating learning curve plots...")
    generate_learning_plots(results_df, title="RL Agent Learning (Lotka-Volterra)")
    
    # Generate timing statistics table
    print("Generating execution time statistics...")
    timing_stats = generate_timing_table(output_filename, max_runs)
    print(f"Timing statistics exported to: execution_time_statistics.csv")
    print(timing_stats)