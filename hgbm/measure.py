from utils import *
import argparse
import os
import inquirer

from utils_measurements import *

if os.name == 'nt':
    dir_str = '\\'
else:
    dir_str = '/'

def get_input_directory(setting_data):
    if bool(setting_data.get('save_timestamp')):
        input_directory = setting_data['output_directory'].split(dir_str)[:-1]
        input_directory = dir_str.join(input_directory)
    else:
        input_directory = setting_data['output_directory']
    subfolders_exists = contains_subdirectory(input_directory)
    if subfolders_exists:
        subdirs = [walker[0] for walker in os.walk(input_directory)]
        questions = [
            inquirer.List('subdir',
                message='Pick a subdirectory',
                choices=subdirs
            ),
        ]
        answer = inquirer.prompt(questions)
        subdir = answer['subdir']
        input_directory = subdir
    return input_directory

def get_simulation_data(setting_data):

    input_directory = setting_data['input_directory']
    verbose = setting_data['verbose']
    if verbose: print_time(f"Input directory: {input_directory}")
    input_path = os.path.join(input_directory, setting_data['input_filename'])
    if (os.path.exists(input_path)) is False:
        raise FileExistsError(f"Input file {input_path} does not exist")
    input_data = np.load(input_path, allow_pickle=True).item()
    n_tests = input_data['n_tests']
    if verbose: print_time(f"Reading output files from {input_directory}")
    p_ij = []
    fitness = []
    theta = []
    if bool(n_tests):
        for test_id in range(n_tests):
            if verbose: print(f"\tTest {test_id + 1} of {n_tests}")
            test_id = test_id
            fitness_file = add_test_id(setting_data['fitness_filename'], test_id + 1)
            theta_file = add_test_id(setting_data['theta_filename'], test_id + 1)
            p_ij_file = add_test_id(setting_data['probability_filename'], test_id + 1)
            fitness_path = os.path.join(input_directory, fitness_file)
            theta_path = os.path.join(input_directory, theta_file)
            p_ij_path = os.path.join(input_directory, p_ij_file)
            if (os.path.exists(fitness_path) and os.path.exists(p_ij_path)) is False:
                raise FileExistsError(f"Output files {fitness_path} AND {p_ij_path} do not exist")
            fitness_sim = np.load(fitness_path)
            p_ij_sim = load_triu(p_ij_path)
            theta_sim = np.load(theta_path)
            p_ij.append(list(p_ij_sim))
            fitness.append(list(fitness_sim))
            theta.append(list(theta_sim))
    else:
        if verbose: print(f"\tTest 1 of 1")
        fitness_file = setting_data['fitness_filename']
        theta_file = setting_data['theta_filename']
        p_ij_file = setting_data['probability_filename']
        fitness_path = os.path.join(input_directory, fitness_file)
        theta_path = os.path.join(input_directory, theta_file)
        p_ij_path = os.path.join(input_directory, p_ij_file)

        if (os.path.exists(fitness_path) and os.path.exists(p_ij_path)) is False:
            raise FileExistsError(f"Output files {fitness_path} AND {p_ij_path} do not exist")
        fitness_sim = np.load(fitness_path)
        p_ij_sim = load_triu(p_ij_path)
        theta_sim = np.load(theta_path)
        p_ij.append(list(p_ij_sim))
        fitness.append(list(fitness_sim))
        theta.append(list(theta_sim))
    p_ij = np.array(p_ij)
    fitness = np.array(fitness)
    theta = np.array(theta)
    
    return p_ij, fitness, theta

def main(config_path):
    # Read data
    setting_data = load_config_file(config_path, mode = 'measurements')
    verbose = setting_data['verbose']
    if verbose: print_time(f"Verbosity activated")
    input_directory = get_input_directory(setting_data)
    setting_data['input_directory'] = input_directory
    output_directory = os.path.join(setting_data['input_directory'], 'measurements')
    input_path = os.path.join(setting_data['input_directory'], setting_data['input_filename'])
    input_data = np.load(input_path, allow_pickle=True).item()
    communities = input_data['communities']
    communities_names = input_data['communities_names']
    delta = input_data['delta']
    N, E, K = expected_measures(communities, delta, input_data['avg_deg'])
    p_ij, fitness, theta = get_simulation_data(setting_data)
    create_folder(output_directory, setting_data['verbose'])

    # # Measure and save results
    if verbose: print_time(f"Measuring degree distribution...")
    measure_and_save_degree_distribuion(p_ij, output_directory, verbose = verbose,avg_deg = input_data.get('avg_deg'), alpha = input_data.get('alpha'), xmin = input_data.get('xmin'))
    if verbose: print_time(f"Measuring K...")
    K_path = os.path.join(output_directory, 'K.png')
    K_sim = measure_K_sim(p_ij, communities, communities_names, verbose = verbose)
    plot_heatmaps(K, K_sim, K_path)
    if verbose: print_time(f'File saved: {K_path}')
    K_err_path = os.path.join(output_directory, 'K_err.png')
    K_err = K_relative_error(K, K_sim)
    matrix_heatmap(K_err, communities_names, K_err_path)
    if verbose: print_time(f'File saved: {K_err_path}')

    if verbose: print_time(f"Measuring E...")
    E_path = os.path.join(output_directory, 'edges.txt')
    E_sim = measure_E_sim(p_ij)
    print(f"Edges:\n\tExpected: {E}\n\tMeasured: {E_sim.mean(axis = 0)} +/- {E_sim.std(axis = 0)}")
    with open(E_path, 'w') as file:
        line = f"Edges:\n\tExpected: {E}\n\tMeasured: {E_sim.mean(axis = 0)} +/- {E_sim.std(axis = 0)}\n"
        file.write(line)
    if verbose: print_time(f" Measuring the giant component...")
    giant_sim_path = os.path.join(output_directory, 'giant_component.txt')
    giant_sim = measure_giant_sim(p_ij, verbose = verbose)
    for test in range(giant_sim.shape[0]):
        print(f"\tTest {test + 1}: {giant_sim[test][0]:.4f} +/- {giant_sim[test][1]:.4f}")
    print(f"Giant component average on tests: {giant_sim.mean(axis = 0)[0]:.4f} +/- {giant_sim.std(axis = 0)[0]:.4f}")
    with open(giant_sim_path, 'w') as file:
        for test in range(giant_sim.shape[0]):
            line = f"\tTest {test + 1}: {giant_sim[test][0]:.4f} +/- {giant_sim[test][1]:.4f}\n"
            file.write(line)
        summary_line = f"Giant component average on tests: {giant_sim.mean(axis=0)[0]:.4f} +/- {giant_sim.std(axis=0)[0]:.4f}\n"
        file.write(summary_line)

    if verbose: print_time(f"Measuring clustering distribution...")
    clustering_data = process_and_plot_clustering(p_ij, output_directory, verbose = verbose)

    if verbose: print_time(f"Measuring average nearest neighbors degree...")
    annd_data = process_and_plot_annd(p_ij, output_directory, verbose = verbose)

    if verbose: print_time(f"Measuring triangles...")
    triangles_sim_path = os.path.join(output_directory, 'triangles.txt')
    triangles_sim = measure_triangles_sim(p_ij, verbose = verbose)
    print(f"Triangles: {triangles_sim}")
    with open(triangles_sim_path, 'w') as file:
        line = f"Triangles: {triangles_sim}\n"
        for test in range(triangles_sim.shape[0]):
            line = f"\tTest {test + 1}: {triangles_sim[test][0]:.4f}\n"
            file.write(line)
        summary_line = f"Triangles average on tests: {triangles_sim.mean():.4f} +/- {triangles_sim.std():.4f}\n"
        file.write(summary_line)

    if verbose: print_time(f"Measuring assortativity...")
    assortativity_sim_path = os.path.join(output_directory, 'assortativity.txt')
    assortativity_sim = measure_assortativity_sim(p_ij, verbose = verbose)
    for test in range(assortativity_sim.shape[0]):
        print(f"\tTest {test + 1}: {assortativity_sim[test][0]:.4f} +/- {assortativity_sim[test][1]:.4f}")
    print(f"Giant component average on tests: {assortativity_sim.mean(axis = 0)[0]:.4f} +/- {assortativity_sim.std(axis = 0)[0]:.4f}")
    with open(assortativity_sim_path, 'w') as file:
        for test in range(assortativity_sim.shape[0]):
            line = f"\tTest {test + 1}: {assortativity_sim[test][0]:.4f} +/- {assortativity_sim[test][1]:.4f}\n"
            file.write(line)
        summary_line = f"Giant component average on tests: {assortativity_sim.mean(axis=0)[0]:.4f} +/- {assortativity_sim.std(axis=0)[0]:.4f}\n"
        file.write(summary_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.py', help='path to config file')
    args = parser.parse_args()
    main(args.config)


#--Removed functions---
    # fitness_vs_degree_path = fitness_path = os.path.join(output_directory, 'degree_vs_fitness.png')
    # fitness_vs_degree(degrees_sim, fitness_sim, fitness_vs_degree_path)
    # if verbose: print_time(f'File saved: {degree_path}')
    
    # fitness_path = os.path.join(output_directory, 'fitness.png')
    # measure_fitness(fitness_sim, input_data.get('avg_deg'), input_data.get('alpha'), fitness_path, input_data.get('xmin'))
    # if verbose: print_time(f'File saved: {fitness_path}')

    # clustering_path = os.path.join(output_directory, 'clustering.png')
    # measure_clustering(clustering_sim, clustering_path)
    # if verbose: print_time(f'File saved: {clustering_path}')

    # K_err_path = os.path.join(output_directory, 'K_err.png')
    # K_std_path = os.path.join(output_directory, 'K_std.png')
    # K_err = K_relative_error(K, K_sim)
    # K_std = K_standard_devs(K, K_sim)
    # matrix_heatmap(K_err, communities_names, K_err_path)
    # if verbose: print_time(f'File saved: {K_err_path}')
    # if np.any(K_std == 0) is False:
    #     matrix_heatmap(K_std, communities_names, K_std_path, vlim = (None, None))
    #     if verbose: print_time(f'File saved: {K_std_path}')
