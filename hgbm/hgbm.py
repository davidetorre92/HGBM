from utils import *
from utils_measurements import measure_K_sim
import argparse
import os
from datetime import datetime

def do_the_tests(input_data, out_folder_path, test_id = None):
    fitness, theta, p_ij = sample_graph_from_formatted_input(input_data)
    fitness_path = os.path.join(out_folder_path, add_test_id(input_data['fitness_filename'], test_id))
    theta_path = os.path.join(out_folder_path, add_test_id(input_data['theta_filename'], test_id))
    p_ij_path = os.path.join(out_folder_path, add_test_id(input_data['probability_filename'], test_id))
    if input_data['verbose']: 
        print_time("Done with sample_graph_from_formatted_input")
        print_time(f"Measured expected edges: {p_ij.sum() / 2}")
        print_time(f"Measured probability\n{measure_K_sim(p_ij, input_data['communities'], input_data['communities_names'])}")

    save_triu(p_ij_path, p_ij)
    np.save(fitness_path, fitness)
    print_time(f"fitness saved to {fitness_path}")
    np.save(theta_path, theta)
    print_time(f"theta saved to {theta_path}")

def main(config_path):
    input_data = load_config_file(config_path)
    if input_data['verbose']: start_time = datetime.now()
    out_folder_path = input_data['output_directory']
    input_path = os.path.join(out_folder_path, input_data['input_filename'])
    create_folder(out_folder_path, input_data['verbose'])
    np.save(input_path, input_data)
    
    if bool(input_data['n_tests']):
        for test_id in range(input_data['n_tests']):
            do_the_tests(input_data, out_folder_path, test_id = test_id + 1)
    else:
        do_the_tests(input_data, out_folder_path, test_id = None)
        
    if input_data['verbose']: end_time = datetime.now()
    if input_data['verbose']: print_time(f"Elapsed time: {end_time - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.py', help='path to config file')
    args = parser.parse_args()
    main(args.config)