import argparse
import importlib.util
from utils_ensamble import *

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config.py file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Now you can access the variables like this:
    ensamble_id_format_spec = config.ensamble_id_format_spec
    output_directory = config.output_directory
    simulation_df_path = config.simulation_df_path
    ensamble_parameters_paths = config.ensamble_parameters_paths
    ensamble_paths_loader = config.ensamble_paths_loader
    options_metadata_loader = config.options_metadata_loader
    options_loader = config.options_loader
    original_graph_path = getattr(config, 'original_graph_path', None)
    ensamble_paths_loader_kwargs = getattr(config, 'ensamble_paths_loader_kwargs', {})
    options_metadata_loader_kwargs = getattr(config, 'options_metadata_loader_kwargs', {})
    options_loader_kwargs = getattr(config, 'options_loader_kwargs', {})
    verbose = getattr(config, 'verbose', False)
    
    if verbose:
        print("------- Settings -------")
        print(f"\tensamble_id_format_spec: {ensamble_id_format_spec}")
        print(f"\toutput_directory: {output_directory}")
        print(f"\tsimulation_df_path: {simulation_df_path}")
        print(f"\toriginal_graph_path: {original_graph_path}")
        print(f"\tensamble_parameters_paths: {ensamble_parameters_paths}")
        print()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        if verbose: print(f"Created output directory: {output_directory}")
    else:
        if verbose: print(f"Output directory already exists: {output_directory}")

    simulation_df = {'parameters': [], 'n_ensambles': [], 'measure': [], 'path_to_measure': [], 'options_path': []}
    simulation_parameters = np.load(ensamble_parameters_paths[0][1], allow_pickle = True).item()

    if verbose:
        print("------- Measurements starts -------")
    for param_id, (ensamble_dir, option_path) in enumerate(ensamble_parameters_paths):
        ensamble_paths = ensamble_paths_loader(ensamble_dir, **ensamble_paths_loader_kwargs)
        options_metadata = options_metadata_loader(option_path, **options_metadata_loader_kwargs)
        options = options_loader(option_path, **options_metadata_loader_kwargs)
        param_metadata = '_'.join([f'{key}_{val}' for key, val in options_metadata.items()])
        if verbose: print(f"Working on {param_id + 1} / {len(ensamble_parameters_paths)} simulations: {options_metadata}")                
        # # Global measures
        # simulation_df = measure_and_save(simulation_df, get_global_df, options_metadata, ensamble_paths, param_metadata, output_directory, option_path, 'Global', {'verbose': verbose, 'n_ensamble': len(ensamble_paths)})
        # # Clustering 
        # simulation_df = measure_and_save(simulation_df, get_clustering_df, options_metadata, ensamble_paths, param_metadata, output_directory, option_path, 'Clustering', {'verbose': verbose, 'n_ensamble': len(ensamble_paths)})
        # Samplings 
        # simulation_df = measure_and_save(simulation_df, get_samplings_df, options_metadata, ensamble_paths, param_metadata, output_directory, option_path, 'Samplings', {'verbose': verbose, 'n_ensamble': len(ensamble_paths)})
        # Degree data
        # simulation_df = measure_and_save(simulation_df, get_degree_df, options_metadata, ensamble_paths, param_metadata, output_directory, option_path, 'Degree')
        # # Triangles
        # simulation_df = measure_and_save(simulation_df, get_triangles_df, options_metadata, ensamble_paths, param_metadata, output_directory, option_path, 'Triangles')
        # # ANND
        # simulation_df = measure_and_save(simulation_df, get_average_neighbor_degree_df, options_metadata, ensamble_paths, param_metadata, output_directory, option_path, 'Average Neighbor Degree')
        # K
        simulation_df = measure_and_save(simulation_df, get_K_df, options_metadata, ensamble_paths, param_metadata, output_directory, option_path, 'K', get_df_kwargs = {'communities': options['communities'], 'communities_names': options['communities_names']})

    simulation_df = pd.DataFrame(simulation_df)
    simulation_df.to_pickle(simulation_df_path)