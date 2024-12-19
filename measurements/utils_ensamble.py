import numpy as np
import pandas as pd
import igraph as ig
from multiprocessing import Pool
import os

ensamble_id_format_spec = '02d'

def get_hgbm_ensamble(ensamble_dir, file_name = 'hgbm'):
    ensamble_files = []
    walker = os.walk(ensamble_dir)
    for dir, _, files in walker:
        open_up = [files[i] for i in range(len(files)) if file_name in files[i]]
        if len(open_up) > 0:
            ensamble_files.extend([os.path.join(dir, file) for file in open_up])

    return ensamble_files

def get_hgbm_options(options_path):
    options = np.load(options_path, allow_pickle=True).item()
    return_dict = {'beta': options['beta']}
    return return_dict

def load_upper_triangle(ensamble_path):
    upper_triangle_values = np.load(ensamble_path)

    # Evaluate number of nodes
    N = int(np.sqrt(upper_triangle_values.size * 2))
    reconstructed_matrix = np.zeros((N, N))

    # Get the indices of the upper triangle
    triu_indices = np.triu_indices(N)

    # Place the values back into the reconstructed matrix
    reconstructed_matrix[triu_indices] = upper_triangle_values

    # Since the matrix is symmetrical, copy the upper triangle to the lower triangle
    reconstructed_matrix = reconstructed_matrix + reconstructed_matrix.T - np.diag(np.diag(reconstructed_matrix))
    return reconstructed_matrix

def measure_K_sim(p_ij, communities, communities_names, verbose = False):
    if len(p_ij.shape) > 2: # There are multiple simulations
        K_sim = []
        for n_test in range(p_ij.shape[0]):
            if verbose: print(f'\tWorking on simulation {n_test + 1} / {p_ij.shape[0]}...')
            K_this_sim = measure_K_sim(p_ij[n_test,:,:], communities, communities_names)
            K_sim.append(K_this_sim)
        K_sim = np.array(K_sim)
    else:    
        n_communities = len(communities_names)
        K_sim = np.zeros((n_communities, n_communities))
        for i, comm_i in enumerate(communities_names):
            comm_i_indices = communities[comm_i]
            for j, comm_j in enumerate(communities_names):
                comm_j_indices = communities[comm_j]
                rows = p_ij[comm_i_indices,:]
                rows_and_cols = rows[:,comm_j_indices]
                K_sim[i,j] += np.sum(rows_and_cols.ravel())
    return K_sim

def get_clustering_data(ensamble_path, n_samples = 100, **kwargs):
    # Get the parameters
    verbose = kwargs.get('verbose', False)
    n_ensamble = kwargs.get('n_ensamble', 0)
    i_ensamble = kwargs.get('i_ensamble', 0) + 1
    # Preparare data
    upper_triangle_values = np.load(ensamble_path)
    N = int(np.sqrt(upper_triangle_values.size * 2))
    triu_indices = np.triu_indices(N)
    # Prepare arguments for parallel execution
    args_list = [
        (
            i_sample,
            n_samples,
            verbose,
            upper_triangle_values,
            triu_indices,
            N,
            i_ensamble,
            n_ensamble
        )
        for i_sample in range(n_samples)
    ]
    # Evaluate
    with Pool(4) as pool:
        results = pool.map(process_sample_clustering, args_list)
    # Unpack results
    degree_data = np.array([res[0] for res in results])
    local_clustering_coefficient_data = np.array([res[1] for res in results])
    degrees = np.unique(degree_data)
    avg_local_clustering_coefficient = np.array([np.mean(local_clustering_coefficient_data[degree_data == degree]) for degree in degrees])
    std_local_clustering_coefficient = np.array([np.std(local_clustering_coefficient_data[degree_data == degree]) for degree in degrees])

    return degrees, avg_local_clustering_coefficient, std_local_clustering_coefficient

def process_sample_clustering(args):
    
    (
        i_sample,
        n_samples,
        verbose,
        upper_triangle_values,
        triu_indices,
        N,
        i_ensamble,
        n_ensamble
    ) = args

    if verbose:
        ending_string = f'{i_ensamble}/{n_ensamble}' if n_ensamble > 0 else '' 
        print('[' + '-'*(i_sample+1) + ' '*(n_samples-i_sample-1) + ']' + ending_string, end='\r')

    samples = np.random.rand(upper_triangle_values.size)
    mask = samples < upper_triangle_values
    edges = list(zip(triu_indices[0][mask], triu_indices[1][mask]))

    g = ig.Graph(n=N, edges=edges, directed=False)
    avg_local_clustering_coefficient = np.array(g.transitivity_local_undirected())
    degree = np.array(g.degree())
    return degree, avg_local_clustering_coefficient

def get_clustering_df(ensamble_paths, **data_kwargs):
    clustering_df = {'ensamble_id': [], 'node_degree': [], 
                     'avg_local_clustering_coefficient': [], 'std_local_clustering_coefficient': []}
    for ensamble_id, ensamble_path in enumerate(ensamble_paths):
        data_kwargs['i_ensamble'] = ensamble_id
        degrees, avg_local_clustering_coefficient, std_local_clustering_coefficient = get_clustering_data(ensamble_path, **data_kwargs)
        clustering_df['ensamble_id'].append(f'{ensamble_id:{ensamble_id_format_spec}}')
        clustering_df['node_degree'].append(degrees)
        clustering_df['avg_local_clustering_coefficient'].append(avg_local_clustering_coefficient)
        clustering_df['std_local_clustering_coefficient'].append(std_local_clustering_coefficient)

    clustering_df = pd.DataFrame(clustering_df)
    return clustering_df

def get_triangles_data(ensamble_path):
    reconstructed_matrix = load_upper_triangle(ensamble_path)
    cubed_matrix = np.linalg.matrix_power(reconstructed_matrix, 3)
    triangles = np.diagonal(cubed_matrix)
    return triangles / 2

def get_triangles_df(ensamble_paths):
    triangles_df = {'ensamble_id': [], 'triangles': []}
    for ensamble_id, ensamble_path in enumerate(ensamble_paths):
        triangles = get_triangles_data(ensamble_path)
        triangles_df['ensamble_id'].append(f'{ensamble_id:{ensamble_id_format_spec}}')
        triangles_df['triangles'].append(triangles)
    triangles_df = pd.DataFrame(triangles_df)
    return triangles_df

def get_average_neighbor_degree_vector(ensamble_path):
    reconstructed_matrix = load_upper_triangle(ensamble_path)
    degrees = np.sum(reconstructed_matrix, axis = 0)
    average_neighbor_degree = np.zeros(reconstructed_matrix.shape[0])
    for i in range(reconstructed_matrix.shape[0]):
        weights_i = reconstructed_matrix[i]
        average_neighbor_degree[i] = np.sum(degrees * weights_i) / np.sum(weights_i)
    return degrees, average_neighbor_degree

def get_average_neighbor_degree_df(ensamble_paths):
    average_neighbor_degree_df = {'ensamble_id': [], 'degrees': [], 'average_neighbor_degree': []}
    for ensamble_id, ensamble_path in enumerate(ensamble_paths):
        degrees, average_neighbor_degree = get_average_neighbor_degree_vector(ensamble_path)
        average_neighbor_degree_df['ensamble_id'].append(f'{ensamble_id:{ensamble_id_format_spec}}')
        average_neighbor_degree_df['average_neighbor_degree'].append(average_neighbor_degree)
        average_neighbor_degree_df['degrees'].append(degrees)
    average_neighbor_degree_df = pd.DataFrame(average_neighbor_degree_df)
    return average_neighbor_degree_df

def get_degree_vector(ensamble_path):
    reconstructed_matrix = load_upper_triangle(ensamble_path)
    degrees = np.sum(reconstructed_matrix, axis = 0)
    return degrees

def get_degree_df(ensamble_paths):
    degree_df = {'ensamble_id': [], 'degree_vector': []}
    for ensamble_id, ensamble_path in enumerate(ensamble_paths):
        degrees = get_degree_vector(ensamble_path)
        degree_df['ensamble_id'].append(f'{ensamble_id:{ensamble_id_format_spec}}')
        degree_df['degree_vector'].append(degrees)
    degree_df = pd.DataFrame(degree_df)
    return degree_df

def get_global_df(ensamble_paths, n_samples = 100, **data_kwargs):
    global_df = {'ensamble_id': [], 'assortativity_avg': [], 'assortativity_std': [],
                 'giant_component_avg': [], 'giant_component_std': [], 
                 'clustering_coefficients_avg': [], 'clustering_coefficients_std': []}
    verbose = data_kwargs.get('verbose', False)
    for ensamble_id, ensamble_path in enumerate(ensamble_paths):
        upper_triangle = np.load(ensamble_path)
        N = int(np.sqrt(upper_triangle.size * 2))
        triu_indices = np.triu_indices(N)
        args = [(i, n_samples, verbose, upper_triangle, triu_indices, N, ensamble_id, len(ensamble_paths)) for i in range(n_samples)]
        with Pool(4) as pool:
            results = pool.starmap(process_sample_global_data, args)

        assortativity_measures = [res[0] for res in results]
        giant_component = [res[1] for res in results]
        clustering_coefficients = [res[2] for res in results]
        global_df['ensamble_id'].append(f'{ensamble_id:{ensamble_id_format_spec}}')
        global_df['assortativity_avg'].append(np.mean(assortativity_measures))
        global_df['assortativity_std'].append(np.std(assortativity_measures))
        global_df['giant_component_avg'].append(np.mean(giant_component))
        global_df['giant_component_std'].append(np.std(giant_component))
        global_df['clustering_coefficients_avg'].append(np.mean(clustering_coefficients))
        global_df['clustering_coefficients_std'].append(np.std(clustering_coefficients))
    return pd.DataFrame(global_df)

def process_sample_global_data(args):
    i_sample, n_samples, verbose, upper_triangle, triu_indices, N, i_ensamble, n_ensamble = args
    if verbose:
        ending_string = f'{i_ensamble}/{n_ensamble}' if n_ensamble > 0 else '' 
        print('[' + '-'*(i_sample+1) + ' '*(n_samples-i_sample-1) + ']' + ending_string, end='\r')
    sample = np.random.rand(upper_triangle.size)
    mask = sample < upper_triangle
    edges = list(zip(triu_indices[0][mask], triu_indices[1][mask]))
    g = ig.Graph(n=N, edges=edges, directed=False)
    assortativity_measures = g.assortativity_degree(directed = False)
    giant_component = g.components().giant().vcount() / g.vcount()
    clustering_coefficients = g.transitivity_avglocal_undirected()
    return assortativity_measures, giant_component, clustering_coefficients

def get_samplings_df(ensamble_paths, **data_kwargs):
    global_df = {'ensamble_id': [], 'node_degree': [], 
                     'avg_local_clustering_coefficient': [], 'std_local_clustering_coefficient': [],
                     'assortativity_avg': [], 'assortativity_std': [],
                     'giant_component_avg': [], 'giant_component_std': [],
                     'clustering_coefficients_avg': [], 'clustering_coefficients_std': []}
    for ensamble_id, ensamble_path in enumerate(ensamble_paths):
        data_kwargs['i_ensamble'] = ensamble_id
        degree_data, local_clustering_coefficient_data, assortativity_measures, giant_component, clustering_coefficients = get_samplings_data(ensamble_path, **data_kwargs)
        degrees = np.unique(degree_data)
        avg_local_clustering_coefficient = np.array([np.mean(local_clustering_coefficient_data[degree_data == degree]) for degree in degrees])
        std_local_clustering_coefficient = np.array([np.std(local_clustering_coefficient_data[degree_data == degree]) for degree in degrees])
        global_df['ensamble_id'].append(f'{ensamble_id:{ensamble_id_format_spec}}')
        global_df['node_degree'].append(degrees)
        global_df['avg_local_clustering_coefficient'].append(avg_local_clustering_coefficient)
        global_df['std_local_clustering_coefficient'].append(std_local_clustering_coefficient)
        global_df['assortativity_avg'].append(np.mean(assortativity_measures))
        global_df['assortativity_std'].append(np.std(assortativity_measures))
        global_df['giant_component_avg'].append(np.mean(giant_component))
        global_df['giant_component_std'].append(np.std(giant_component))
        global_df['clustering_coefficients_avg'].append(np.mean(clustering_coefficients))
        global_df['clustering_coefficients_std'].append(np.std(clustering_coefficients))
    global_df = pd.DataFrame(global_df)
    return global_df

def get_samplings_data(ensamble_path, n_samples = 100, **data_kwargs):
    verbose = data_kwargs.get('verbose', False)
    upper_triangle = np.load(ensamble_path)
    N = int(np.sqrt(upper_triangle.size * 2))
    triu_indices = np.triu_indices(N)
    i_ensamble = data_kwargs.get('i_ensamble', 0)
    n_ensamble = data_kwargs.get('n_ensamble', 0)
    if verbose:
        print('\tPreparing data for the sampling...', end='\r')
    args = [(upper_triangle, triu_indices, N) for i in range(n_samples)]
    if verbose:
        print('\tSampling...                                                            ', end='\r')
    progress = 0
    if verbose:
        i_sample = 0
        ending_string = f'{i_ensamble+1}/{n_ensamble}' if n_ensamble > 0 else '' 
        print('[' + '-'*(i_ensamble+1) + ' '*(n_ensamble - i_ensamble - 1) + ']         ' + ending_string, end='\r')
        progress += 1
    n_parallel_processes = 4
    with Pool(n_parallel_processes) as pool:
        results = pool.map(process_sample, args)

    degree_data = np.array([res[0] for res in results])
    local_clustering_coefficient_data = np.array([res[1] for res in results])
    assortativity_measures = np.array([res[2] for res in results])
    giant_component = np.array([res[3] for res in results])
    clustering_coefficients = np.array([res[4] for res in results])

    return degree_data, local_clustering_coefficient_data, assortativity_measures, giant_component, clustering_coefficients

def process_sample(args):
    upper_triangle, triu_indices, N = args
    sample = np.random.rand(upper_triangle.size)
    mask = sample < upper_triangle
    edges = list(zip(triu_indices[0][mask], triu_indices[1][mask]))
    g = ig.Graph(n=N, edges=edges, directed=False)
    degree_data = np.array(g.degree())
    local_clustering_coefficient_data = np.array(g.transitivity_local_undirected())
    assortativity_measures = g.assortativity_degree(directed = False)
    giant_component = g.components().giant().vcount() / g.vcount()
    clustering_coefficients = g.transitivity_avglocal_undirected()
    return degree_data, local_clustering_coefficient_data, assortativity_measures, giant_component, clustering_coefficients

def get_K_df(ensamble_paths, **kwargs):
    K_df = {'ensamble_id': [], 'K': []}
    communities = kwargs['communities']
    communities_names = kwargs['communities_names']
    for ensamble_id, ensamble_path in enumerate(ensamble_paths):
        p_ij = load_upper_triangle(ensamble_path)
        K = measure_K_sim(p_ij, communities, communities_names)
        K_df['ensamble_id'].append(f'{ensamble_id:{ensamble_id_format_spec}}')
        K_df['K'].append(K)
    return pd.DataFrame(K_df)

def measure_and_save(simulation_df, get_df, options, ensamble_paths, param_metadata, output_dir, option_path, data_str, get_df_kwargs = {}):
    df = get_df(ensamble_paths, **get_df_kwargs)
    df_path = f"{output_dir}/{'_'.join(data_str.lower().split(' '))}_{param_metadata}.pickle"
    df.to_pickle(df_path)
    print(f"{data_str} data saved on {df_path}")
    simulation_df['parameters'].append(options)
    simulation_df['n_ensambles'].append(len(ensamble_paths))
    simulation_df['measure'].append(data_str)
    simulation_df['path_to_measure'].append(df_path)
    simulation_df['options_path'].append(option_path)
    return simulation_df