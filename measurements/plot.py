import argparse
import importlib.util
from utils_ensamble import *
from utils_plot import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.size": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 30
})
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
    if getattr(config, 'output_directory_plot', None) is None: 
        output_directory = config.output_directory + '/figures_and_tables'
    else:
        output_directory = config.output_directory_plot
    simulation_df_path = config.simulation_df_path
    ensamble_parameters_paths = config.ensamble_parameters_paths

    verbose = getattr(config, 'verbose', False)
    original_graph_path = getattr(config, 'original_graph_path', None)
    g_benchmark_function = getattr(config, 'g_benchmark_function', None)
    g_benchmark_kwargs = getattr(config, 'g_benchmark_kwargs', {})

    if verbose:
        print("------- Settings -------")
        print(f"\tensamble_id_format_spec: {ensamble_id_format_spec}")
        print(f"\toutput_directory: {output_directory}")
        print(f"\tsimulation_df_path: {simulation_df_path}")
        print(f"\toriginal_graph_path: {original_graph_path}")
        print(f"\tensamble_parameters_paths: {ensamble_parameters_paths}")
        print(f"\tg_benchmark_function: {g_benchmark_function}")
        print(f"\tg_benchmark_kwargs: {g_benchmark_kwargs}")
        print()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        if verbose: print(f"Created output directory: {output_directory}")
    else:
        if verbose: print(f"Output directory already exists: {output_directory}")

    simulation_df = pd.read_pickle(simulation_df_path)
    if original_graph_path is not None:
        if original_graph_path.endswith('.graphml'):
            G = ig.Graph.Read_GraphML(original_graph_path)
        elif original_graph_path.endswith('.pickle'):
            G = ig.Graph.Read_Pickle(original_graph_path)
        else:
            raise ValueError(f"Unknown graph format: {original_graph_path}")
    else:
        G = None
    if g_benchmark_function is not None:
        g_bm = g_benchmark_function(**g_benchmark_kwargs)
        benchmark_label_string = g_benchmark_kwargs.get('label_string', 'Benchmark')
    else:
        g_bm = None
        benchmark_label_string = None

    if verbose:
        print("------- Plot starts here -------")

    # plot_reconstructed_matrix_extended(G, simulation_df, output_directory, verbose = True, file_name = 'mixing_matrix_real_vs_reconstructed.pdf', flat = True, width = 0.02)

    plot_reconstructed_matrix_v2(G, simulation_df, output_directory, verbose = True, flat = True)
    plot_k_err(G, simulation_df, output_directory, verbose = verbose)
    plot_k(G, simulation_df, output_directory, verbose = verbose)    
    plot_degree(G, simulation_df, output_directory, g_bm = g_bm, benchmark_label_string = benchmark_label_string, verbose = verbose)
    plot_clustering(G, simulation_df, output_directory, g_bm = g_bm, benchmark_label_string = benchmark_label_string, verbose = verbose)
    if G is not None:
        plot_triangles(G, simulation_df, output_directory, verbose = verbose)    
        plot_annd(G, simulation_df, output_directory, verbose = verbose)

    plot_knn(G, simulation_df, output_directory, verbose = verbose)
    plot_global_metrics(G, simulation_df, output_directory, verbose = verbose)
