from utils_plot import stochastic_block_model
import numpy as np
import os
def get_hgbm_ensamble(ensamble_dir, file_name = 'hgbm', **kwargs):
    ensamble_files = []
    walker = os.walk(ensamble_dir)
    for dir, _, files in walker:
        open_up = [files[i] for i in range(len(files)) if file_name in files[i]]
        if len(open_up) > 0:
            ensamble_files.extend([os.path.join(dir, file) for file in open_up])

    return ensamble_files

def get_hgbm_options_randomizer(options_path):
    options = np.load(options_path, allow_pickle=True).item()
    return_dict = {'beta': options['beta']}
    return return_dict

def get_dmerc_options(options_path):
    options = np.load(options_path, allow_pickle=True).item()
    return_dict = {'D': options['dim']}
    return return_dict

def get_hgbm_options_synthetic_article(options_path):
    options = np.load(options_path, allow_pickle=True).item()
    return_dict = {'beta': options['beta']}
    delta = options['delta']
    if delta[0,0] == 3:
        return_dict['diagonal'] = 'strong'
    else:
        return_dict['diagonal'] = 'weak'
    return return_dict


def get_hgbm_parameters(options_path):
    options = np.load(options_path, allow_pickle=True).item()
    return options

def get_dmerc_parameters(options_path):
    options = np.load(options_path, allow_pickle=True).item()
    return options

# Settings
ensamble_id_format_spec = '02d'
output_directory = '/home/davide/AI/Projects/HGBM/article_data/d-mercator'
output_directory_plot = '/home/davide/AI/Projects/HGBM/article_results/d-mercator'
simulation_df_path = f'{output_directory}/simulation_df.pickle'
ensamble_parameters_paths = [
    ('/home/davide/AI/Projects/HGBM/article_data/d-mercator','/home/davide/AI/Projects/HGBM/article_data/dmerc_out/input_file.npy'),
]
# Measures
ensamble_paths_loader = get_hgbm_ensamble
options_metadata_loader = get_dmerc_options
options_loader = get_dmerc_parameters
verbose = True
original_graph_path = '/home/davide/AI/Projects/HGBM/graphs/hgbm_strong_diagonal.graphml'
g_benchmark_function = None
g_benchmark_kwargs = {'settings_path': ensamble_parameters_paths[0][1], 'G_path': original_graph_path, 'label_string' : r'$G_{SBM}$'}