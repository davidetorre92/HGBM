from utils import *
import argparse
import os
from datetime import datetime
import networkx as nx

def main(config_path):
    input_data = load_config_file(config_path)
    if input_data['verbose']: start_time = datetime.now()
    out_folder_path = input_data['output_directory']
    input_path = os.path.join(out_folder_path, input_data['input_filename'])
    create_folder(out_folder_path, input_data['verbose'])
    np.save(input_path, input_data)
    kappas = {i: v for i, v in enumerate(input_data['degrees'])}
    if bool(input_data['n_tests']):
        for test_id in range(input_data['n_tests']):
            G = nx.geometric_soft_configuration_graph(beta = input_data['beta'], kappas = kappas)
            p_ij = nx.to_numpy_array(G)
            p_ij_path = os.path.join(out_folder_path, add_test_id(input_data['probability_filename'], test_id + 1))
            save_triu(p_ij_path, p_ij)
            print_time(f"p_ij saved to {p_ij_path}")
            fitness = np.array(input_data['degrees'])
            fitness_path = os.path.join(out_folder_path, add_test_id(input_data['fitness_filename'], test_id + 1))
            np.save(fitness_path, fitness)
            print_time(f"fitness saved to {fitness_path}")
    else:
        G = nx.geometric_soft_configuration_graph(beta = input_data['beta'], kappas = kappas)
        p_ij = np.array(nx.adjacency_matrix(G).toarray())
        p_ij_path = os.path.join(out_folder_path, input_data['probability_filename'])
        save_triu(p_ij_path, p_ij)
        print_time(f"p_ij saved to {p_ij_path}")
        fitness = np.array(input_data['degrees'])
        fitness_path = os.path.join(out_folder_path, input_data['fitness_filename'])
        np.save(fitness_path, fitness)
        print_time(f"fitness saved to {fitness_path}")
        
    if input_data['verbose']: end_time = datetime.now()
    if input_data['verbose']: print_time(f"Elapsed time: {end_time - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.py', help='path to config file')
    args = parser.parse_args()
    main(args.config)