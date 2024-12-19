import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
from utils import print_time, find_xmin_from_expected_value, expected_measures
import itertools
import igraph as ig
import os
import pdb

def log_smoothing(n):
    if n <= 0:
        return 0
    exp = int(np.log10(n))
    base = int(n / 10**exp)
    return base * 10**exp

def sample_graph_from_probability_matrix(p_ij):
    N = p_ij.shape[0]
    S, T = np.triu_indices(N, k = 1)
    sample = np.random.random(N*(N-1) // 2)
    mask = sample < p_ij[S, T]
    S_mask = S[mask]
    T_mask = T[mask]
    edges = list(zip(S_mask, T_mask))
    g = ig.Graph(n=N, edges=edges, directed=False)
    return g
def measure_log_distribution(data, x_name, bins = None):
    if bins is None:
        max = int(np.max(data)+2)
        bins = [n for n in range(1,max)]
    hist = []
    df_dict = {x_name: [], 'freq': [], 'test_id': []}
    if len(data.shape) > 1: # if I've got more than one measurement
        for i_datum, datum in enumerate(data):
            this_hist, bin_edges = np.histogram(datum, bins = bins, density = True)
            hist.append(list(this_hist))
            df_dict[x_name].extend(list(bin_edges[:-1]))
            df_dict['freq'].extend(list(this_hist))
            df_dict['test_id'].extend([i_datum]*len(this_hist))

    else:
        this_hist, bin_edges = np.histogram(data, bins = bins, density = True)
        hist.append(list(this_hist))
        df_dict[x_name].extend(list(bin_edges[:-1]))
        df_dict['freq'].extend(list(this_hist))
        df_dict['test_id'].extend([0]*len(this_hist))

    df = pd.DataFrame(df_dict)
    return df

def plot_log_distribution(df, x_name, alpha = None, xm = None, pwfit = None, grid = True, title_str = None, bins = None, data_str = 'Data', data_color = 'blue', ax = None, double_log = True):

    hist_avg_df = df.groupby(x_name)['freq'].mean()
    hist_avg = hist_avg_df.values
    avg_bins = hist_avg_df.index.values
    hist_std_df = df.groupby(x_name)['freq'].std()
    hist_std = hist_std_df.values
    std_bins = hist_std_df.index.values
    if (np.all(avg_bins == std_bins)) is False:
        raise ValueError("Something is wrong with the bins!")
    bins = avg_bins
    if ax is None:
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot()
    else:
        fig = ax.get_figure()
    ax.errorbar(bins, hist_avg, yerr=hist_std, label = data_str, color = data_color, fmt = 'o', ecolor = [0, 0, 0.5, 0.5], elinewidth=2, capsize=6)
    ax.set_xscale('log')
    if double_log: ax.set_yscale('log')
    if title_str is not None:
        ax.set_title(title_str)

    if alpha is not None and xm is not None:
        x_data = np.logspace(np.log10(xm),np.log10(bins[-1]),num = 20)
        y = lambda x: np.power(xm * (alpha-2) / x, alpha-1) / x
        y_data = y(x_data)
        ax.plot(x_data, y_data, color = 'red', label = 'Powerlaw distribution' + '\n' + r'$\alpha =$'+f'{alpha:.2f}' + '\n' + r'$x_m=$' + f'{xm:.2f}')

    if grid:
        ax.grid(True)
    return fig, ax

def measure_fitness(fitness, avg_deg, alpha, path, xmin = None):
    # Fit exponents with power law
    # results = powerlaw.Fit(fitness.ravel())
    # print_time(f'Measured exponent of the fitness: {results.power_law.alpha}')
    # print_time(f'Expected exponent of the fitness: {alpha}')
    # if alpha is not None: print_time(f'Relative error: {(alpha - results.power_law.alpha) / alpha * 100:.4f}%')
    # print_time(f'Measured average: {np.mean(fitness)}')
    # print_time(f'Expected average: {avg_deg}')
    # print_time(f'Relative error: {(avg_deg - np.mean(fitness)) / avg_deg * 100:.4f}%')
    # print_time(f'Measured x min of the fitness: {results.power_law.xmin}')
    if bool(xmin):
        xmin = find_xmin_from_expected_value(avg_deg, alpha)
        print_time(f'Expected x min of the fitness: {xmin}')
        # print_time(f'Relative error: {(xmin - results.power_law.xmin) / xmin * 100:.4f}%')


    fig, ax = plot_log_distribution(fitness, alpha = alpha, xm = xmin, pwfit = None, title_str = 'Fitness distribution')
    ax.axvline(avg_deg, label = f'Expected average {avg_deg:.2f}', color = 'red', ls='--')
    ax.axvline(np.mean(fitness), label = f'Measured average {np.mean(fitness):.2f}', color = 'black', ls='--')
    ax.legend(loc = 'upper left', bbox_to_anchor = (1.05, 1.0))
    fig.tight_layout()
    fig.savefig(path)

def measure_and_save_degree_distribuion(p_ij, output_dir, verbose = False, avg_deg = None, alpha = None, xmin = None, fig_name = 'degree.png', df_name = 'degree.pickle'):
    if bool(xmin):
        xmin = find_xmin_from_expected_value(avg_deg, alpha)
        print_time(f'Expected x min of the degree distribution: {xmin}')

    degree = p_ij.sum(axis = 1)
    bins = np.logspace(np.log10(1),np.log10(np.max(degree)),num = 50)
    x_name = 'degree'
    df_degree_data = measure_log_distribution(degree, x_name, bins = bins)
    if len(degree.shape) > 1: 
        pwfit = powerlaw.Fit(degree.ravel(), xmin = xmin)
    else:
        pwfit = powerlaw.Fit(degree, xmin = xmin)
    fig, ax = plot_log_distribution(df_degree_data, x_name, alpha = pwfit.power_law.alpha, xm = pwfit.power_law.xmin, title_str = 'Degree distribution')
    ax.set_ylim((1e-10,None))
    if avg_deg is not None: ax.axvline(avg_deg, label = f'Expected average {avg_deg:.2f}', color = 'red', ls='--')
    ax.axvline(np.mean(degree), label = f'Measured average {np.mean(degree):.2f}', color = 'black', ls='--')
    ax.legend(loc = 'upper left', bbox_to_anchor = (1.05, 1.0), prop = {'size': 16})
    fig.tight_layout()
    image_path = os.path.join(output_dir, fig_name)
    fig.savefig(image_path)
    if verbose: print_time(f'File saved: {image_path}')
    dataframe_path = os.path.join(output_dir, df_name)
    df_degree_data.to_pickle(dataframe_path)
    if verbose: print_time(f'File saved: {dataframe_path}')

    return df_degree_data

def process_and_plot_clustering(p_ij, output_dir, image_name = 'clustering.png', dataframe_name = 'clustering.pickle', smooth = True, verbose = False):
    if len(p_ij.shape) == 2:
        p_ij = p_ij.reshape(1, p_ij.shape[0], p_ij.shape[1])
    clustering_data = measure_clustering_vs_k_class_sim(p_ij, verbose = verbose)
    clustering_data = np.nan_to_num(clustering_data, nan = 0)
    clustering_df = {'degree_class': [], 'mean': [], 'std': [], 'sample_id': [], 'test_id': []}
    for i_test in range(clustering_data.shape[0]):
        for i_sample in range(clustering_data[i_test].shape[0]):
            clustering_df['degree_class'].extend(clustering_data[i_test,i_sample,0])
            clustering_df['mean'].extend(clustering_data[i_test,i_sample,1])
            clustering_df['std'].extend(clustering_data[i_test,i_sample,2])
            clustering_df['sample_id'].extend([i_sample] * len(clustering_data[i_test,i_sample,0]))
            clustering_df['test_id'].extend([i_test] * len(clustering_data[i_test,i_sample,0]))
    clustering_df = pd.DataFrame(clustering_df)
    path = os.path.join(output_dir, dataframe_name)
    clustering_df.to_pickle(path)
    if verbose: print_time(f'File saved: {path}')
    # means = clustering_data[:,:,1,:].mean(axis = 1)
    # stds = clustering_data[:,:,2,:].mean(axis = 1)
    # degrees = clustering_data[0,0,0,:]
    fig, ax = plt.subplots(figsize = (16,9))
    if smooth:
        degrees = clustering_df['degree_class'].unique()
        degrees = {deg: log_smoothing(deg) for deg in degrees}
        clustering_df['degree_class'] = clustering_df['degree_class'].map(degrees)
    degrees = clustering_df['degree_class'].unique()
    means = clustering_df.groupby('degree_class')['mean'].mean()
    stds = clustering_df.groupby('degree_class')['mean'].std()
    degrees = clustering_df['degree_class'].unique()
    if len(means.shape) > 1:
        for i in range(means.shape[0]):
            ax.errorbar(degrees, means[i], yerr = stds[i], label = f'Clustering coefficient test {i+1}', fmt='o', linewidth=2, capsize=6, alpha = 0.5)
    else:
        ax.errorbar(degrees, means, yerr = stds, label = 'Clustering coefficient', fmt='o', linewidth=2, capsize=6, alpha = 0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    ax.set_title("Clustering spectrum")
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\bar{c}(k)$')
    ax.legend(loc = 'upper left', bbox_to_anchor = (1.05, 1.0))
    fig.tight_layout()
    path = os.path.join(output_dir, image_name)
    fig.savefig(path)
    if verbose: print_time(f'File saved: {path}')
    return clustering_data

def process_and_plot_annd(p_ij, output_directory, image_name = 'annd.png', dataframe_name = 'annd.pickle', smooth = True, verbose = False):
    if len(p_ij.shape) == 2:
        p_ij = p_ij.reshape(1, p_ij.shape[0], p_ij.shape[1])
    annd_data = measure_annd_vs_k_class_sim(p_ij, verbose = verbose)
    annd_data = np.nan_to_num(annd_data, nan = 0)
    annd_df = {'degree_class': [], 'mean': [], 'sample_id': [], 'test_id': []}
    for i_test in range(annd_data.shape[0]):
        for i_sample in range(annd_data[i_test].shape[0]):
            annd_df['degree_class'].extend(annd_data[i_test,i_sample,0])
            annd_df['mean'].extend(annd_data[i_test,i_sample,1])
            annd_df['sample_id'].extend([i_sample] * len(annd_data[i_test,i_sample,0]))
            annd_df['test_id'].extend([i_test] * len(annd_data[i_test,i_sample,0]))
    annd_df = pd.DataFrame(annd_df)
    path = os.path.join(output_directory, dataframe_name)
    annd_df.to_pickle(path)
    if verbose: print_time(f'File saved: {path}')
    # means = annd_data[:,:,1,:].mean(axis = 1)
    # stds = annd_data[:,:,1,:].std(axis = 1)
    # degrees = annd_data[0,0,0,:]
    if smooth:
        degrees = annd_df['degree_class'].unique()
        degrees = {deg: log_smoothing(deg) for deg in degrees}
        annd_df['degree_class'] = annd_df['degree_class'].map(degrees)
    degrees = annd_df['degree_class'].unique()
    means = annd_df.groupby('degree_class')['mean'].mean()
    stds = annd_df.groupby('degree_class')['mean'].std()
    fig, ax = plt.subplots(figsize = (16,9))
    if len(means.shape) > 1:
        for i in range(means.shape[0]):
            ax.errorbar(degrees, means[i], yerr = stds[i], label = f'ANND test {i+1}', fmt='o', linewidth=2, capsize=6, alpha = 0.5)
    else:
        ax.errorbar(degrees, means, yerr = stds, label = 'ANND', fmt='o', linewidth=2, capsize=6, alpha = 0.5)
    ax.set_xscale('log')
    ax.grid()
    ax.set_title("Average Nearest Neighbors Degree")
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\bar{k}_{nn}(k)$')
    ax.legend(loc = 'upper left', bbox_to_anchor = (1.05, 1.0))
    fig.tight_layout()
    path = os.path.join(output_directory, image_name)
    fig.savefig(path)
    if verbose: print_time(f'File saved: {path}')
    return annd_data

def fitness_vs_degree(degree, fitness, path):
    if degree.shape != fitness.shape: # if I've got more than one measurement
        print("Degree and fitness should have the same shape")
        return
    if len(degree.shape) > 1:
        degree = degree.mean(axis = 0)
        fitness = fitness.mean(axis = 0)
    fig, axs = plt.subplots(1,2, figsize = (16,5))
    fig, axs[0] = plot_log_distribution(degree, title_str = 'Fitness and degree distribution', ax = axs[0], data_color = 'cyan', data_str = 'Degree Data', grid = True)
    fig, axs[0] = plot_log_distribution(fitness, ax = axs[0], data_color = 'blue', data_str = 'Fitness Data')
    fitness_degree_relative_error = (degree - fitness) / fitness
    axs[1].scatter(fitness, fitness_degree_relative_error, color = 'blue')
    axs[1].set_title('Degree - Fitness relative error distribution')
    axs[1].set_xscale('log')
    axs[1].grid()
    print(f'Degree - Fitness: Mean absolute error = {np.mean(np.abs((degree - fitness))):.4f}')
    fig.savefig(path)

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

def measure_triangles_sim(p_ij, verbose = False):
    if len(p_ij.shape) > 2: # There are multiple simulations
        triangles_sim = []
        for n_test in range(p_ij.shape[0]):
            if verbose: print(f'\tWorking on simulation {n_test + 1} / {p_ij.shape[0]}...')
            triangles_this_sim = measure_triangles_sim(p_ij[n_test,:,:])
            triangles_sim.append(triangles_this_sim)
        triangles_sim = np.array(triangles_sim)
    else:
        p_ij_3 = np.linalg.matrix_power(p_ij, 3)
        triangles_sim = np.array([np.trace(p_ij_3) / 6])
    return triangles_sim

def measure_giant_sim(p_ij, sample_size = 100, verbose = False):
    if len(p_ij.shape) > 2: # There are multiple simulations
        giant_sim = []
        for n_test in range(p_ij.shape[0]):
            if verbose: print(f'\tWorking on simulation {n_test + 1} / {p_ij.shape[0]}...')
            giant_this_sim = measure_giant_sim(p_ij[n_test,:,:])
            giant_sim.append(giant_this_sim)
        giant_sim = np.array(giant_sim)
    else:
        giant_sim_samples = []
        for i_sample in range(sample_size):
            g = sample_graph_from_probability_matrix(p_ij)
            giant_sim = g.clusters().giant().vcount() / g.vcount()
            giant_sim_samples.append(giant_sim)
        giant_sim_samples = np.array(giant_sim_samples)
        giant_sim = [np.mean(giant_sim_samples), np.std(giant_sim_samples)]
    return giant_sim

def measure_assortativity_sim(p_ij, sample_size = 100, verbose = False):
    if len(p_ij.shape) > 2: # There are multiple simulations
        assortativity_sim = []
        for n_test in range(p_ij.shape[0]):
            if verbose: print(f'\tWorking on simulation {n_test + 1} / {p_ij.shape[0]}...')
            assortativity_this_sim = measure_assortativity_sim(p_ij[n_test,:,:])
            assortativity_sim.append(assortativity_this_sim)
        assortativity_sim = np.array(assortativity_sim)
    else:
        assortativity_sim_samples = []
        for i_sample in range(sample_size):
            g = sample_graph_from_probability_matrix(p_ij)
            assortativity_sample = g.assortativity_degree(directed = False)
            assortativity_sim_samples.append(assortativity_sample)
        assortativity_sim_samples = np.array(assortativity_sim_samples)
        assortativity_sim = [np.mean(assortativity_sim_samples), np.std(assortativity_sim_samples)]
    return assortativity_sim

def measure_E_sim(p_ij):
    if len(p_ij.shape) > 2: # There are multiple simulations
        E_sim = []
        for n_test in range(p_ij.shape[0]):
            E_this_sim = measure_E_sim(p_ij[n_test,:,:])
            E_sim.append(E_this_sim)
        E_sim = np.array(E_sim)
    else:    
        E_sim = np.array([np.sum(p_ij) / 2])
    return E_sim
def _measure_clustering_sim(p_ij):
    # pdb.set_trace()
    N = p_ij.shape[0]
    k = np.sum(p_ij > 0, axis = 1)
    s = np.sum(p_ij, axis = 1)
    c = np.zeros(N)
    for i in range(N):
        if k[i] == 1:
            c[i] = 0
            continue
        indices = [k for k in range(0,i)] + [k for k in range(i+1,N)]
        indices = list(itertools.combinations(indices, 2))
        c[i] = 1 / (s[i] * (k[i]-1)) * 2 * np.sum([p_ij[i,h]*p_ij[h,k]*p_ij[k,i] for (h,k) in indices])
    c = np.nan_to_num(c, nan = 0.0)
    return c

def measure_clustering_vs_k_class_sim(p_ij, sample_size = 100, verbose = False):
    def get_c_func_k(clustering, degrees, k, func = np.mean):
        return func(clustering[degrees == k])
    
    if len(p_ij.shape) > 2: # There are multiple simulations
        c_sim = []
        for n_test in range(p_ij.shape[0]):
            if verbose: print(f'\tWorking on simulation {n_test + 1} / {p_ij.shape[0]}...')
            c_this_sim = measure_clustering_vs_k_class_sim(p_ij[n_test,:,:], sample_size = sample_size, verbose = verbose)
            c_sim.append(c_this_sim)
        c_sim = np.array(c_sim)
    else:
        c_sim_samples = []
        degree_class = [k for k in range(2,p_ij.shape[0])]
        for i_sample in range(sample_size):
            g = sample_graph_from_probability_matrix(p_ij)
            clustering = g.transitivity_local_undirected()
            clustering = np.nan_to_num(clustering, nan = 0.0)
            degrees = np.array(g.degree())
            c_avg_k = [get_c_func_k(clustering, degrees, k, func = np.mean) for k in degree_class]
            c_std_k = [get_c_func_k(clustering, degrees, k, func = np.std) for k in degree_class]
            c_sim_samples.append([degree_class, c_avg_k, c_std_k])
        c_sim = np.array(c_sim_samples)
    return c_sim

def measure_annd_vs_k_class_sim(p_ij, sample_size = 100, verbose = False):
    if len(p_ij.shape) > 2: # There are multiple simulations
        annd_sim = []
        for n_test in range(p_ij.shape[0]):
            if verbose: print(f'\tWorking on simulation {n_test + 1} / {p_ij.shape[0]}...')
            annd_this_sim = measure_annd_vs_k_class_sim(p_ij[n_test,:,:], sample_size = sample_size, verbose = verbose)
            annd_sim.append(annd_this_sim)
        annd_sim = np.array(annd_sim)
    else:
        annd_sim_samples = []
        degree_class = [k for k in range(1, p_ij.shape[0])]
        for i_sample in range(sample_size):
            g = sample_graph_from_probability_matrix(p_ij)
            annd = np.nan_to_num(g.knn()[1], 0)
            annd = {i+1: annd[i] for i in range(len(annd))}
            annd_sim = [annd.get(k,0) for k in degree_class]
            annd_sim_samples.append([degree_class, annd_sim]) # annd_sim_samples
        annd_sim = np.array(annd_sim_samples)
    return annd_sim

def K_relative_error(K, K_sim):
    if len(K_sim.shape) == 3:
        K_sim_avg = np.mean(K_sim, axis = 0)
        relative_error_K = (K_sim_avg - K) / K_sim_avg
    else:
        relative_error_K = (K_sim - K) / K_sim
    relative_error_K = np.nan_to_num(relative_error_K, nan = 0)
    return relative_error_K

def K_standard_devs(K, K_sim):
    if len(K_sim.shape) == 3:
        K_sim_avg = np.mean(K_sim, axis = 0)
        K_sim_std = np.std(K_sim, axis = 0)
        relative_error_K = (K_sim_avg - K) / K_sim_std
    else:
        relative_error_K = (K_sim - K) / K
    relative_error_K = np.nan_to_num(relative_error_K, nan = 0)
    return relative_error_K

def matrix_heatmap(matrix, communities_names, path, matrix_text = None, vlim = (-0.1, 0.1)):
    fig, ax = plt.subplots(figsize=(6, 5))
    vmin, vmax = vlim
    # colormap
    cmap = 'coolwarm'
    # Plot heatmaps
    im1 = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='none', vmin = vmin, vmax = vmax)

    # Set titles
    ax.set_title('Relative error matrix')

    # Add color bar
    cbar = fig.colorbar(im1, ax=ax, orientation='vertical')
    cbar.set_label('Relative error')

    if matrix_text is None:
        matrix_text = [[f'{matrix[i, j]:.2f}' for j in range(matrix.shape[1])] for i in range(matrix.shape[0])]

    # Show values in the heatmap boxes
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix_text[i][j], ha='center', va='center', color='w')
    

    ax.set_xticks(np.arange(len(communities_names)), labels=communities_names)
    ax.set_yticks(np.arange(len(communities_names)), labels=communities_names)
    fig.savefig(path)
    return fig, ax

def edges_error(A, N_blocks, delta, avg_deg):
    _, E, _ = expected_measures(N_blocks, delta, avg_deg)
    E_sim = np.sum(A) / 2
    print(f"Expected edges: {E}")
    print(f"Edges simulation: {E_sim}")
    print(f"Relative error edges: {(E_sim - E) / E * 100}%")

import matplotlib.ticker as ticker

def plot_heatmaps(matrix1, matrix2, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})

    # Using 'plasma' colormap
    cmap = 'YlGn'

    # Check if the matrix contains multiple measurements
    if len(matrix2.shape) == 3:
        matrix2_avg = np.mean(matrix2, axis = 0)
        matrix2_std = np.std(matrix2, axis = 0)

    else:
        matrix2_avg = matrix2
        matrix2_std = np.zeros(matrix2.shape) 

    # Plot heatmaps
    im1 = ax1.imshow(matrix1, cmap=cmap, aspect='auto', interpolation='none')
    im2 = ax2.imshow(matrix2_avg, cmap=cmap, aspect='auto', interpolation='none')

    # Set titles
    ax1.set_title('Input matrix')
    ax2.set_title('Output from simulation')

    # Add color bar
    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', location = 'bottom', aspect = 60)
    cbar.set_label('Value')

    # Show values in the heatmap boxes
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            ax1.text(j, i, f'{matrix1[i, j]:.2f}', ha='center', va='center', color='black')

    for i in range(matrix2_avg.shape[0]):
        for j in range(matrix2_avg.shape[1]):
            if len(matrix2.shape) == 3:
                if matrix2.shape[0] > 1: ax2.text(j, i, f'{matrix2_avg[i, j]:.2f}\n({matrix2_std[i, j]:.2f})', ha='center', va='center', color='black', size = 7)
                else: ax2.text(j, i, f'{matrix2_avg[i, j]:.2f}', ha='center', va='center', color='black', size = 7)
            else:
                ax2.text(j, i, f'{matrix2_avg[i, j]:.2f}', ha='center', va='center', color='black')

    # Adjust layout
    ax1.xaxis.set_major_locator(ticker.NullLocator())
    ax1.yaxis.set_major_locator(ticker.NullLocator())
    ax2.xaxis.set_major_locator(ticker.NullLocator())
    ax2.yaxis.set_major_locator(ticker.NullLocator())

    fig.savefig(path)
    return

from matplotlib.pyplot import cm
def degree_distribution(degree, communities, path):
    n_communities = len(communities.keys())
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    color = cm.rainbow(np.linspace(0, 1, n_communities))
    for community_id, (community, node_indices) in enumerate(communities.items()):
        stubs_block = degree[node_indices]
        mean = np.mean(stubs_block)
        plot_log_distribution(stubs_block, data_str = f'Degree group {community}', ax = axs, data_color = color[community_id])
        axs.axvline(mean, label = f'Mean degree group {community_id}: {mean:.2f}', c = color[community_id])
        axs.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        # plot_log_distribution(stubs_block, data_str = f'Degree group {group_id}', data_color = 'blue', ax = axs[group_id])
        # axs[group_id].axvline(mean, label = f'Mean degree group {group_id}: {mean:.2f}', color = 'red')
        # axs[group_id].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.suptitle("Degree distributions per group")
    fig.savefig(path)

def get_dist(data):
    data = sorted(data)
    xdata = []
    ydata = []
    x = data[0]
    y = 1
    i = 1
    l = len(data)
    while(i<l):
        xnew = data[i]
        if xnew == x:
            y += 1
        else:
            xdata.append(x)
            ydata.append(y/l)
            x = xnew
            y = 1
        i += 1
    xdata.append(x)
    ydata.append(y/l)
    return xdata, ydata


def get_fit_dist(alpha, xmin, xmax, scale):
    xfit = np.linspace(xmin,xmax,num=100)
    beta = (alpha-1)/(xmin**(1-alpha)-xmax**(1-alpha)) * scale
    yfit = beta*xfit**(-alpha)
    return xfit, yfit


def fit_data(data, xmin=None, xmax=None, compare=False, plot=False, discrete=True, output_file='', verbose=False):
    ''' fit data to a power law

        Args:
            data            is the distribution to fit, can be either a list or a dictionary
                            if a list, is interpreted as the list of all outcomes (e.g., [0,1,2,1,1] means pr(0)=pr(2)=0.2, pr(1)=0.6)
                            if a dictionary, items are interpreted as outcome:number_of_occurrences (e.g., {0:1, 1:3, 2:1})
            xmin
            xmax
            compare         another distribution to which the power law fit can be compared; if True or invalid, exponential is used
            plot            specifies whether the fit must be plotted (both pdf fit and ccdf fit are plotted)
            output_file     the output file for the plot; if empty a default name is used
    '''
    if isinstance(data,dict):
        d = []
        for k,v in data.items():
            d += [k]*v
        data = np.array(d)
    else:
        data = np.array(data)

    fit = powerlaw.Fit(data=data, discrete=discrete, xmin=xmin, xmax=xmax, linear_bins=True)
    if verbose:
        print('Power-law best fit to data:')
        print('\t alpha = {}'.format(fit.power_law.alpha))
        print('\t sigma = {}'.format(fit.power_law.sigma))
        print('\t xmin = {}'.format(fit.power_law.xmin))
        print('\t xmax = {}'.format(fit.power_law.xmax))
    if compare:
        if compare == True:
            R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio = True)
        else:
            try:
                R, p = fit.distribution_compare('power_law', compare, normalized_ratio=True)
            except:
                print('the argument to option "compare" is not a valid distribution, will use exponential')
                R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio = True)
        if verbose:
            print('Normalized loglikelihood ratio: {}'.format(R))
            print('Significance value: {}'.format(p))
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10,10))
        fit.plot_pdf(color= 'b', ax=ax, label='data pdf')
        fit.power_law.plot_pdf(color= 'b',linestyle='--',label='fit pdf', ax=ax)
        fit.plot_ccdf(color='r', ax=ax, label='data ccdf')
        fit.power_law.plot_ccdf(color='r', linestyle='--', ax=ax, label='fit ccdf')
        plt.legend(loc = 'upper left', bbox_to_anchor = (1.05, 1.0))
        if not output_file:
            output_file = 'power_law_fit.png'
        fig.savefig(output_file)
    return fit

def plot_degree_distribution_with_fit(data, plot_to=None):

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    for k,d in data.items():
        #### scatter plot of data distribution ####
        d = np.array(d)
        xdata, ydata = get_dist(d)

        if xdata[0]==0:
            prob0 = '={:.3f})'.format(ydata[0])
        else:
            prob0 = '=0)'

        ax.scatter(xdata, ydata, s=80, alpha=0.4, label=f'{k} (' + r'$\Pr[0]$' + prob0)
        ymindata = min(ydata)

        fit = fit_data(d, discrete=True, compare=True, verbose=False)
        alpha = fit.alpha
        sigma = fit.sigma
        D     = fit.D
        xmin  = fit.xmin
        xmax = xdata[-1]
        scale = sum(ydata[xdata.index(xmin):xdata.index(xmax)])
        xfit, yfit = get_fit_dist(alpha, xmin, xmax, scale)

        xfit = xfit[yfit>=ymindata]
        yfit = yfit[yfit>=ymindata]
        ax.plot(xfit, yfit, '--', label=f'PL fit of {k} (' + r'$\alpha$' +'={:.2f})'.format(alpha))

        print("\n")
        print("-"*80)
        print(f"FIT DATA for {k}:")
        print("-"*80)
        print("alpha = {}, sigma = {}, D = {}".format(alpha, sigma, D))
        print("xmin = {}, xmax = {}".format(xmin, xmax))
        print("-"*80)
        print("\n")

    title = 'degree distribution'
    ax.set_title(title, fontsize=10)
    ax.legend(loc = 'upper left', bbox_to_anchor = (1.05, 1.0), prop={'size': 12})
    ax.tick_params(labelsize=18)
    ax.set_ylabel('Probability', labelpad=40, fontsize=20)

    if plot_to is None:
        outfile = 'degree_distribution.png'
    else:
        outfile = plot_to
    fig.savefig(outfile, bbox_inches='tight') #, dpi=150)
    print("plot_distribution_with_fit: plot saved to {}".format(outfile))

    return

