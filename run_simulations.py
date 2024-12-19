import os
import itertools
import numpy as np
import sys  
sys.path.append('/home/davide/AI/Projects/HGBM')

# Base directory where results will be saved
import argparse
import datetime
import pdb

def is_line_in_file(file_path, line_to_check):
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == line_to_check.strip():
                return True
    return False

# Parameter ranges
N_values = [5000]
C_values = [4]
diagonal_modes = ['Strong', 'Weak']
beta_values = [0.1, 0.5]
alpha_values = [2.5]
avg_deg_values = [10.0]
community_attributes = [None]
graph_paths = [None]
adjust_hidden_degrees = [True]
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=int, default=0, help='Writing mode: 0 = append, 1 = overwrite')
args = parser.parse_args()

mode = args.mode
base_output_dir = "/home/davide/AI/Projects/HGBM/article_data/adjusted_degrees"

# N_values = [200]
# C_values = [2]
# diagonal_modes = ['Strong']
# beta_values = [10.0]
# alpha_values = [2.5]
# avg_deg_values = [10.0]

# Prepare the log file
log_file_path = os.path.join(base_output_dir, 'simulation_log.txt')
os.makedirs(base_output_dir, exist_ok=True)

mode = 'a' if mode == 0 else 'w'
if mode == 'w':
    input('You selected overwrite mode. This mode will overwrite all previous results. Press 1 to confirm:')
if (mode == 'a' and os.path.exists(log_file_path) is False) or (mode == 'w'):
    with open(log_file_path, mode) as log_file:
        log_file.write("Simulation Log\n")
        log_file.write("====================\n\n")

# Generate all combinations
combinations = list(itertools.product(N_values, C_values, diagonal_modes, beta_values, alpha_values, avg_deg_values, community_attributes, graph_paths, adjust_hidden_degrees))

for idx, (N, C, diag_mode, beta, alpha, avg_deg, community_attribute, graph_path, adjust_flag) in enumerate(combinations, start=1):
    id_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set up the communities
    communities = {}
    if C is not None:
        community_size = N // C
        communities = {str(i): [n for n in range(i * community_size, (i + 1) * community_size)] for i in range(C)}
    configuration_string = f"N={N}, C={C}, diagonal_mode={diag_mode}, beta={beta}, alpha={alpha}, avg_deg={avg_deg} community_attribute={community_attribute}, graph_path={graph_path}, adjust_hidden_degrees={adjust_flag}\n"

    if mode == 'a':
        if is_line_in_file(log_file_path, configuration_string):
            print(f"Skipping configuration {id_str}: {configuration_string}")
            continue
    # Set up the delta matrix
    if C is not None:
        if diag_mode == 'Strong':
            delta = [[3 if i == j else 1 for j in range(C)] for i in range(C)]
        else:  # Weak diagonal
            delta = [[1.2 if i == j else 1 for j in range(C)] for i in range(C)]
    else:
        delta = [[]]
    # Create the output directory for this combination
    output_dir = os.path.join(base_output_dir, f"simulation_{id_str}")
    os.makedirs(output_dir, exist_ok=True)

    # Create the config dictionary
    config = {
        "communities": communities,
        "delta": delta,
        "degrees": None,
        "beta": beta,
        "alpha": alpha,
        "avg_deg": avg_deg,
        "xmin": None,
        "verbose": True,
        "graph_path": graph_path,
        "community_attribute": community_attribute,
        "output_directory": output_dir,
        "save_timestamp": False,
        "n_tests": 10,
        "adjust_hidden_degrees": adjust_flag
    }

    # Save the config to a config.py file
    config_file_path = os.path.join(output_dir, 'config.py')
    with open(config_file_path, 'w') as config_file:
        config_file.write(f"import numpy as np\n\nconfig = {config}\n")

    # Document the configuration
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Simulation {id_str}:\n")
        log_file.write(configuration_string)
        log_file.write(f"Output directory: {output_dir}\n\n")

    # Run the simulations
    os.system(f"python3 hgbm/hgbm.py -c {config_file_path}")
    # os.system(f"python3 hgbm/measure.py -c {config_file_path}")

print(f"All simulations completed. Log file saved at {log_file_path}.")
