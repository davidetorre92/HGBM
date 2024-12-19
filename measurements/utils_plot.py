import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import copy
import powerlaw
import os
import seaborn as sns

def stochastic_block_model(**kwargs):
    settings_path = kwargs['settings_path']
    G_path = kwargs['G_path']
    G = ig.Graph.Read_GraphML(G_path)
    V = G.vcount()
    parameters = np.load(settings_path, allow_pickle = True).item()
    block_sizes = [len(c) for c in parameters['communities'].values()]
    delta = parameters['delta']
    K = delta / np.sum(delta) * 2 * G.ecount()
    pref_matrix = np.zeros(K.shape)
    for i in range(pref_matrix.shape[0]):
        for j in range(pref_matrix.shape[1]):
            if i == j:
                p = block_sizes[i] * (block_sizes[j] - 1) // 2
            else:
                p = block_sizes[i] * block_sizes[j]
            pref_matrix[i, j] = K[i][j] / p *0.5 / 10.75 * 11.75

    g_sbm = ig.Graph.SBM(V, pref_matrix, block_sizes)
    return g_sbm

def label_constructor(options):
    string_list = []
    for key, value in options.items():
        if key == 'beta':
            string_list.append(r'$\beta$' + f' = {value}')
        elif key == 'D':
            string_list.append(f'D = {value}') 
        elif key == 'Diagonal':
            if value == 'Strong':
                value = '3 / 1'
                string_list.append(f'K ratio = {value}')
            elif value == 'Very strong':
                value = '6 / 1'
                string_list.append(f'K ratio = {value}')
            elif value == 'Weak':
                value = '1.2 / 1'
                string_list.append(f'K ratio = {value}')
            else:
                pass
        elif key == 'diagonal':
            pass
        else:
            string_list.append(f'{key.capitalize()}: {value}')
        
    return '\n'.join(string_list)

def area_plot(x, y, std, avg = None, ax = None, label = None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    line, = ax.plot(x, y, '-', label=label)
    ax.fill_between(x, y - 2*std, y + 2*std, alpha=0.2, color=line.get_color())
    if avg is not None:
        ax.axvline(x = avg, color = line.get_color(), linestyle = (0, (5, 5)), alpha = .5)
    ax.legend()
    return ax

def plot_degree_from_ensambles_distribution(degree_path, options, ax = None, bins = None, fit = False):
    degree_df = pd.read_pickle(degree_path)
    label_string = label_constructor(options)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if bins is None:
        deg_min = 1
        deg_max = degree_df['degree_vector'].map(max).max()
        bins = np.logspace(np.log10(deg_min), np.log10(deg_max), 20)
        
    ensambles = degree_df.shape[0]
    n_bins = len(bins) - 1
    histogram_data = np.zeros((ensambles, n_bins))
    for i, degree_vector in enumerate(degree_df['degree_vector']):
        hist_this_data = np.histogram(degree_vector, bins = bins, density = False)[0]
        histogram_data[i, :] = 1 - np.cumsum(hist_this_data) / np.sum(hist_this_data)
    if fit:
        degrees = degree_df['degree_vector'][0]
        fit = powerlaw.Fit(degrees)

    hist_avg = histogram_data.mean(axis = 0)
    hist_avg /= hist_avg[0]
    hist_std = histogram_data.std(axis = 0)
    avg = np.mean(np.concatenate(degree_df['degree_vector']))
    ax = area_plot(bins[:-1], hist_avg, hist_std, ax = ax, avg = avg, label = label_string)
    if fit:
        fit.power_law.plot_ccdf(ax = ax, label = r'PW Fit: $\alpha$' + f' = {fit.power_law.alpha:.2f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$P_c(k)$')
    ax.set_title(f'Degree distribution')
    return ax

def plot_degree_from_graph(G, ax = None, bins = None, label_string = 'Empirical data'):
    degrees = np.array(G.degree())
    avg_deg = np.mean(degrees)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if bins is None:
        deg_min = 1
        deg_max = np.max(degrees)
        bins = np.logspace(np.log10(deg_min), np.log10(deg_max), 20)
    
    histogram_data = np.histogram(degrees, bins = bins, density = False)[0]
    cumulative_hist_avg = np.cumsum(histogram_data) / np.sum(histogram_data)
    ccdf = 1 - cumulative_hist_avg
    ccdf /= ccdf[0]
    line, = ax.plot(bins[:-1], ccdf, 'o-', label=label_string)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$P_c(k)$')
    ax.set_title(f'Degree distribution')
    ax.legend()
    ax.axvline(x = avg_deg, color = line.get_color(), linestyle = (0, (5, 5)))

def plot_clustering_from_ensambles_distribution(ensamble_path, options, ax = None, fit = True):
    # Collecting results for each measured degree
    df = pd.read_pickle(ensamble_path)
    label_string = label_constructor(options)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    deg_min = 2
    deg_max = df['node_degree'].map(max).max()
    data = {deg: [] for deg in range(deg_min, int(deg_max + 1))}
    # Iterate over the ensambles
    for row_id, row in df.iterrows():
        # Collect measured degree of the ensamble
        degrees = row['node_degree']
        # ... and its clustering coefficient
        lcc = row['avg_local_clustering_coefficient']
        for deg in degrees:
            # Skip clustering coefficient for 0 and 1: avoid useless nans.
            if deg == 0 or deg == 1: continue
            mask = np.argwhere(degrees == deg)
            data[deg].extend([lcc[m][0] for m in mask])

    # Set the x and y axis
    x = []
    y = []
    std = []
    for deg, values in data.items():
        # Avoid to plot empty measurements
        if values == []: continue
        x.append(deg)
        y.append(np.mean(values))
        std.append(np.std(values))

    x = np.array(x)
    y = np.array(y)
    std = np.array(std)
    ax = area_plot(x, y, std, avg = None, ax = ax, label = label_string)

    return ax

def plot_clustering_from_graph(G, ax = None, label = 'Empirical data'):
    degrees = np.array(G.degree())
    local_clustering_coefficient = np.array(G.transitivity_local_undirected())
    k = np.unique(degrees)
    avg_local_clustering_coefficient = np.array([np.mean(local_clustering_coefficient[degrees == degree]) for degree in k])
    ax.plot(k, avg_local_clustering_coefficient, 'o-', label = label)
    return ax

def get_triangles_from_G(G):
    degrees = np.array(G.degree())
    lcc = np.array(G.transitivity_local_undirected())
    triangles_opt = lcc * (degrees - 1) * degrees / 2
    return triangles_opt

def plot_triangle_data(ensamble_path, triangles, options, ax = None):
    # Collecting results for each measured degree
    df = pd.read_pickle(ensamble_path)
    label_string = label_constructor(options)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    scatter = None
    for row_id, ensamble_measurement in df.iterrows():
        mask = ~np.isnan(triangles)
        if scatter is None:
            scatter = ax.scatter(triangles[mask], ensamble_measurement['triangles'][mask], label = label_string, alpha=0.05)
            markerfacecolor = scatter.get_facecolor()[0]
        else:
            ax.scatter(triangles[mask], ensamble_measurement['triangles'][mask], color = markerfacecolor)
    
    return scatter

def plot_annd_data(ensamble_path, original_annd, options, ax = None):
    # Collecting results for each measured degree
    df = pd.read_pickle(ensamble_path)
    label_string = label_constructor(options)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    scatter = None
    for row_id, ensamble_measurement in df.iterrows():
        if scatter is None:
            scatter = ax.scatter(original_annd, ensamble_measurement['average_neighbor_degree'], label = label_string, alpha=0.05)
            markerfacecolor = scatter.get_facecolor()[0]
        else:
            ax.scatter(original_annd, ensamble_measurement['average_neighbor_degree'], color = markerfacecolor)
    return scatter

def plot_knn_from_ensambles_distribution(ensamble_path, options, ax = None):
    # Collecting results for each measured degree
    df = pd.read_pickle(ensamble_path)
    label_string = label_constructor(options)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Iterate over the ensambles
    uniq_deg = np.unique(np.concatenate([np.array(row['degrees'][~np.isnan(row['degrees'])], dtype = int) for _, row in df.iterrows()]))
    data = {deg: [] for deg in uniq_deg}
    for row_id, row in df.iterrows():
        # Collect measured degree of the ensamble
        degrees = np.array([int(deg) for deg in row['degrees'][~np.isnan(row['degrees'])]])
        # ... and their annd
        annd = row['average_neighbor_degree']
        uniq_deg = np.unique(degrees)
        for deg in uniq_deg:
            # Skip clustering coefficient for 0 and 1: avoid useless nans.
            if deg == 0: continue
            mask = np.argwhere(degrees == deg)
            data[deg].append(np.mean(annd[mask]))
    
    # Set the x and y axis
    x = []
    y = []
    std = []
    for deg, values in data.items():
        # Avoid to plot empty measurements
        if values == []: continue
        x.append(deg)
        y.append(np.mean(values))
        std.append(np.std(values))
    

    x = np.array(x)
    y = np.array(y)
    std = np.array(std)
    ax = area_plot(x, y, std, ax = ax, avg = None, label = label_string)

    return ax

def plot_knn_from_graph(G, ax = None, label = 'Empirical data'):
    degrees = np.array(G.degree())
    average_neighbor_degree = np.array(G.knn()[0])
    k = np.unique(degrees)
    knn = np.array([np.mean(average_neighbor_degree[degrees == degree]) for degree in k])
    ax.plot(k, knn, 'o-', label = label)
    return ax

def plot_K_ensamble(ensamble_paths, K_real, options, ax = None, vlim = (-0.5,0.5), label_order = None, text_fontsize = None):
    subplot_title = label_constructor(options)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    df = pd.read_pickle(ensamble_paths)
    K = np.array([K for K in df['K']])
    K_avg = np.mean(K, axis = 0)
    if label_order is not None:
        if len(label_order.keys()) != K_avg.shape[0]:
            raise ValueError("The number of labels must be the same as the number of degrees")
        K_avg_ordered = np.zeros(K_avg.shape)
        K_real_ordered = np.zeros(K_real.shape)
        for i in range(K_avg.shape[0]):
            for j in range(K_avg.shape[1]):
                K_avg_ordered[label_order[i], label_order[j]] = K_avg[i, j]
                K_real_ordered[label_order[i], label_order[j]] = K_real[i, j]
        K_avg = K_avg_ordered
        K_real = K_real_ordered
    relative_error = np.divide(K_avg - K_real, K_real, out=np.zeros_like(K_avg), where=K_real!=0)
    print(options)
    print("Average relative error: ", np.mean(np.abs(relative_error)))
    print("Min relative error: ", np.min(np.abs(relative_error)))
    print("Max relative error: ", np.max(np.abs(relative_error)))
    print()
    vmin, vmax = vlim
    # colormap
    cmap = 'coolwarm_r'
    # Plot heatmaps
    im1 = ax.imshow(relative_error, cmap=cmap, aspect='auto', interpolation='none', vmin = vmin, vmax = vmax)

    # Set titles
    ax.set_title(subplot_title)

    # # Add color bar
    # fig.colorbar(im1, ax=[ax], orientation='vertical', shrink=0.8)
    # cbar = fig.axes[-1]
    # cbar.set_label('Relative error')

    matrix_text = [[f'{K_avg[i, j] + 0.5:.0f}\n({K_real[i, j]:.0f})' for j in range(K_avg.shape[1])] for i in range(K_avg.shape[0])]
    if text_fontsize is None:
        text_fontsize = plt.rcParams['font.size']
    # Show values in the heatmap boxes if the number of labels is less than 10
    if K_avg.shape[0] < 10:
        for i in range(K_avg.shape[0]):
            for j in range(K_avg.shape[1]):
                ax.text(j, i, matrix_text[i][j], ha='center', va='center', color='black', fontsize = text_fontsize)

    #Turn off spines to make grid visible
    ax.spines[:].set_visible(False)

    return fig, ax
    
def global_metrics_table_graph(G, index = 'Original'):
    assortativity = G.assortativity_degree()
    giant_component = G.components().giant().vcount() / G.vcount()
    clustering_coefficient = G.transitivity_avglocal_undirected()
    row = {'Assortativity': assortativity, 'Giant Component': giant_component, 'Clustering Coefficient': clustering_coefficient,
           'Options': index}
    return row
           
def global_metrics_table_ensamble(ensamble_paths, options):
    df = pd.read_pickle(ensamble_paths)
    avg = df.mean(numeric_only = True)
    std = df.std(numeric_only = True)
    row = {'Assortativity': avg['assortativity_avg'], 'Assortativity Std': std['assortativity_avg'],
           'Giant Component': avg['giant_component_avg'], 'Giant Component Std': std['giant_component_avg'], 
           'Clustering Coefficient': avg['clustering_coefficients_avg'], 'Clustering Coefficient Std': std['clustering_coefficients_avg'],
           'Options': label_constructor(options)}
    return row

def make_latex_table(df, table_path, fmt = '04f'):
    df_return = {'Options': [], 'Assortativity': [], 'Giant Component': [], 'Clustering Coefficient': []}
    for row_id, row in df.iterrows():
        if pd.isna(row['Assortativity Std']):
            df_return['Options'].append(row['Options'])
            df_return['Assortativity'].append(f"${row['Assortativity']:.{fmt}}$")
            df_return['Giant Component'].append(f"${row['Giant Component']:.{fmt}}")
            df_return['Clustering Coefficient'].append(f"${row['Clustering Coefficient']:.{fmt}}")
    for row_id, row in df.iterrows():
        if pd.isna(row['Assortativity Std']) is False:
            df_return['Options'].append(row['Options'])
            df_return['Assortativity'].append(f"${row['Assortativity']:.{fmt}} \pm {row['Assortativity Std']:.{fmt}}$")
            df_return['Giant Component'].append(f"${row['Giant Component']:.{fmt}} \pm {row['Giant Component Std']:.{fmt}}$")
            df_return['Clustering Coefficient'].append(f"${row['Clustering Coefficient']:.{fmt}} \pm {row['Clustering Coefficient Std']:.{fmt}}$")

    df = pd.DataFrame(df_return)
    df.to_latex(table_path, escape = False, index = False)

def plot_reconstructed_matrix(G, simulation_df, output_directory, verbose = True, file_name = 'mixing_matrix_real_vs_reconstructed.pdf'):
    # Prepare plot
    fig, ax = plt.subplots(figsize = (14, 9))
    # discard first color
    if G is not None:
        ax.set_prop_cycle(color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
    
    for id, experiment_row in simulation_df[simulation_df['measure'] == 'K'].iterrows():
        # Measure ground truth mixing matrix, K_real
        parameters = np.load(experiment_row['options_path'], allow_pickle = True).item()
        delta = 2 * parameters['delta'] / np.sum(parameters['delta'])
        df_path = experiment_row['path_to_measure']
        df = pd.read_pickle(df_path)
        scatter = None
        for id, ensamble_row in df.iterrows():
            K = np.array(ensamble_row['K'])
            delta_rec = K / np.sum(K) * 2
            if scatter is None:
                scatter = ax.scatter(delta.flatten(), delta_rec.flatten(), alpha = 0.5, label = label_constructor(experiment_row['parameters']))
            else:
                ax.scatter(delta.flatten(), delta_rec.flatten(), color = scatter.get_facecolor()[0])
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ax.grid()
    handles, labels = ax.get_legend_handles_labels()
    legend_handles = [copy.copy(h) for h in handles]
    for h in legend_handles:
        h.set_alpha(1)
    ax.legend(legend_handles, labels, loc = 'upper left', bbox_to_anchor = (1, 1.05))
    m = min(minx, miny) * 0.9
    M = max(maxx, maxy) * 1.1
    eps = 0.1
    ax.plot([m,M], [m,M], linestyle = '--', color = 'black')
    ax.fill_between([m, M], [(1 - eps) * m, M * (1 - eps)], [(1 + eps) * m, (1 + eps) * M], color = 'blue', alpha = 0.1)
    ax.set_xlim(m, M)
    ax.set_ylim(m, M)
    fig.suptitle('Mixing matrix reconstruction')
    ax.set_xlabel('Real values')
    ax.set_ylabel('Reconstructed values')
    fig.tight_layout()
    image_path = os.path.join(output_directory, file_name)
    fig.savefig(image_path)
    if verbose:
        print(f"K plot saved to: {image_path}")
    return

def plot_reconstructed_matrix_v2(G, simulation_df, output_directory, verbose = True, file_name = 'mixing_matrix_real_vs_reconstructed.pdf', flat = True, width = 0.02):
    # Prepare plot
    fig, ax = plt.subplots(figsize = (16, 9))
    # discard first color
    if G is not None:
        ax.set_prop_cycle(color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
    rows = []
    label_order = []
    for id, experiment_row in simulation_df[simulation_df['measure'] == 'K'].iterrows():
        # Measure ground truth mixing matrix, K_real
        parameters = np.load(experiment_row['options_path'], allow_pickle = True).item()
        delta = np.round(2 * parameters['delta'] / np.sum(parameters['delta']), 3)
        df_path = experiment_row['path_to_measure']
        df = pd.read_pickle(df_path)
        label_order.append(label_constructor(experiment_row['parameters']))
        for id, ensamble_row in df.iterrows():
            K = np.array(ensamble_row['K'])
            delta_rec = K / np.sum(K) * 2
            for real, rec in zip(delta.flatten(), delta_rec.flatten()):
                rows.append([real, rec, label_constructor(experiment_row['parameters'])])
    plot_df = pd.DataFrame(rows, columns = ['real values', 'reconstructed values', 'label'])
    plot_df.sort_values('real values', ascending = True, inplace = True)
    data_to_n_labels = {x: len(plot_df[plot_df['real values'] == x]['label'].unique().tolist()) for x in plot_df['real values'].unique().tolist()}
    if flat:
        data_to_x = {x: n * width * 2 * max(data_to_n_labels.values()) for n, x in enumerate(plot_df['real values'].unique().tolist())}
        offsets = {x: (- data_to_n_labels[x] // 2) * width / 2 for x in plot_df['real values'].unique().tolist()}
        offsets_increment = width
    else:
        data_to_x = {x: x for x in plot_df['real values'].unique().tolist()}
        offsets = {x: 0 for x in plot_df['real values'].unique().tolist()}
        offsets_increment = 0
    for label in label_order:
        data_avg = plot_df[plot_df['label'] == label].drop('label', axis = 1).groupby('real values').mean()
        data_std = plot_df[plot_df['label'] == label].drop('label', axis = 1).groupby('real values').std()
        X_data = np.array(data_avg.index)
        X_plot = [] # The array containing the x position of each bar
        for x in X_data:
            x_plot = data_to_x[x]
            X_plot.append(x_plot + offsets[x])
            offsets[x] = offsets[x] + offsets_increment
        y = data_avg['reconstructed values']
        y_std = data_std['reconstructed values']
        bars = ax.bar(X_plot, 2 * y_std, bottom = y - y_std, width=width, alpha = 0.5)
        color = bars.patches[0].get_facecolor()
        ax.errorbar(X_plot, y, yerr = 2 * y_std, label = label, color = color, fmt = 'o', linewidth = 2, capsize = 6, alpha = 0.5)

    ax.grid()
    X = plot_df['real values'].unique().tolist()
    if flat:
        ax.errorbar([data_to_x[x] for x in X], X, xerr = [data_to_n_labels[x] * width / 2 for x in X], linestyle = 'none', marker='none', color = 'black', label = 'Input data', zorder=10)
    else:
        ax.scatter([data_to_x[x] for x in X], X, marker='o', s=16, color = 'black', label = 'Input data', zorder=10)

    ax.legend(loc = 'upper left', bbox_to_anchor = (1.0, 1.05))
    if flat:
        ax.set_xticks([data_to_x[x] for x in X])
        ax.set_xticklabels([f'{x:.4f}' for x in X], rotation = 45)
    ax.set_xlabel(r'Input values')
    ax.set_ylabel(r'Reconstructed values')
    ax.set_title(r'Reconstruction of the mixing matrix $\Delta$')
    fig.tight_layout()
    image_path = os.path.join(output_directory, file_name)
    fig.savefig(image_path)
    if verbose:
        print(f"K plot saved to: {image_path}")
    return

def plot_reconstructed_matrix_extended(G, simulation_df, output_directory, verbose = True, file_name = 'mixing_matrix_real_vs_reconstructed.pdf', flat = True, width = 0.02):
    # Prepare plot
    fig, ax = plt.subplots(figsize = (16, 9))
    # discard first color
    if G is not None:
        ax.set_prop_cycle(color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
    rows = []
    label_order = []
    for id, experiment_row in simulation_df[simulation_df['measure'] == 'K'].iterrows():
        # Measure ground truth mixing matrix, K_real
        parameters = np.load(experiment_row['options_path'], allow_pickle = True).item()
        delta = np.round(2 * parameters['delta'] / np.sum(parameters['delta']), 3)
        df_path = experiment_row['path_to_measure']
        df = pd.read_pickle(df_path)
        label_order.append(label_constructor(experiment_row['parameters']))
        upper_triangle_indices = np.triu_indices(delta.shape[0])
        for id, ensamble_row in df.iterrows():
            K = np.array(ensamble_row['K'])
            delta_rec = K / np.sum(K) * 2
            for i_row, i_col in zip(upper_triangle_indices[0], upper_triangle_indices[1]):
                real = delta[i_row, i_col]
                rec = delta_rec[i_row, i_col]
                rows.append([real, rec, i_row, i_col, id, label_constructor(experiment_row['parameters'])])
    plot_df = pd.DataFrame(rows, columns = ['real values', 'reconstructed values', 'i_row', 'i_col', 'id', 'label'])
    print(plot_df)
    sns.scatterplot(data = plot_df, x = 'real values', y = 'reconstructed values', hue = 'label', ax = ax, palette = 'bright')
    # ax.scatter(plot_df['real values'], plot_df['reconstructed values'], label = plot_df['label'],alpha = 0.5)
    fig.savefig(os.path.join(output_directory, file_name))
    if verbose:
        print(f"K plot saved to: {os.path.join(output_directory, file_name)}")
    return


def plot_k_err(G, simulation_df, output_directory, verbose = True, file_name = 'K_error_plot.pdf'):
    # Prepare plot
    fig, ax = plt.subplots(figsize = (16, 9))
    X = []
    y_diag = []
    y_outdiag = []
    y_diag_err = []
    y_outdiag_err = []
    y_diag_true = []
    y_outdiag_true = []
    
    for id, experiment_row in simulation_df[simulation_df['measure'] == 'K'].iterrows():
        # Measure ground truth mixing matrix, K_real
        parameters = np.load(experiment_row['options_path'], allow_pickle = True).item()
        block_sizes = [len(c) for c in parameters['communities'].values()]
        delta = parameters['delta']
        if G is not None: E = G.ecount()
        else:
            avg_deg = parameters['avg_deg']
            N = np.sum(block_sizes)
            E = N * avg_deg / 2
        K_real = delta / np.sum(delta) * 2 * E
        df_path = experiment_row['path_to_measure']
        options_metadata = experiment_row['parameters']
        y_diag_true.append(np.mean(np.diagonal(K_real)))
        y_outdiag_true.append(np.mean(np.array([K_real[i,j] for i in range(len(K_real)) for j in range(len(K_real)) if i != j])))

        df_path = experiment_row['path_to_measure']
        df = pd.read_pickle(df_path)
        scatter = None
        diag_data = []
        outdiag_data = []
        for id, ensamble_row in df.iterrows():
            K = np.array(ensamble_row['K'])
            diag = np.mean(np.diagonal(K))
            outdiag = np.mean(np.array([K[i,j] for i in range(len(K)) for j in range(len(K)) if i != j]))
            diag_data.append(diag)
            outdiag_data.append(outdiag)

        y_diag.append(np.mean(diag_data))
        y_outdiag.append(np.mean(outdiag_data))
        y_diag_err.append(np.std(diag_data))
        y_outdiag_err.append(np.std(outdiag_data))
        X.append(label_constructor(options_metadata))

    ms_value = 10
    marker_true_value = 'd'
    ms_true_value = 200
    error_bar_container = ax.errorbar(X, y_diag, yerr = y_diag_err, linestyle='none', marker = 'o', label = 'Diagonal', ms = ms_value, alpha = 0.75)
    ax.scatter(X, y_diag_true, marker = marker_true_value, label = 'Diagonal true', color = error_bar_container.lines[0].get_color(), s = ms_true_value, alpha = 0.75)
    error_bar_container = ax.errorbar(X, y_outdiag, yerr = y_outdiag_err, linestyle='none', marker = 'o', label = 'Outdiagonal', ms = ms_value, alpha = 0.75)
    ax.scatter(X, y_outdiag_true, marker = marker_true_value, label = 'Outdiagonal true', color = error_bar_container.lines[0].get_color(), s = ms_true_value, alpha = 0.75)
    ax.set_xlabel('Randomization options')
    ax.set_ylabel('Average edge counts')
    ax.grid(axis = 'y')
    ax.legend(loc = 'upper left', bbox_to_anchor=(1.05, 1.0))
    X_tick_pos = ax.get_xticks()
    for i, x in enumerate(X_tick_pos):
        if i % 2 == 0:
            ax.axvspan(x - 0.5, x + 0.5, color = 'lightgrey', alpha = 0.5, zorder = -1)
    ax.tick_params(axis='x', rotation=45)
    fig.suptitle('Edge counts comparison')
    fig.tight_layout()
    image_path = os.path.join(output_directory, file_name)
    fig.savefig(image_path)
    if verbose:
        print(f"Comparison diagonal and off-diagonal: {image_path}")

def plot_k(G, simulation_df, output_directory, verbose = True, file_name = 'K.pdf'):
    # Prepare plot
    max_cols = 2
    n_exp = simulation_df[simulation_df['measure'] == 'K'].shape[0]
    n_cols = n_exp if n_exp < max_cols else max_cols
    n_rows = n_exp // max_cols
    if n_exp % max_cols != 0:
        n_rows += 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize = (7 * n_cols, 7 * n_rows), layout = 'constrained', sharex=True, sharey=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    i_plot = 0
    for id, experiment_row in simulation_df[simulation_df['measure'] == 'K'].iterrows():
        # Measure ground truth mixing matrix, K_real
        parameters = np.load(experiment_row['options_path'], allow_pickle = True).item()
        block_sizes = [len(c) for c in parameters['communities'].values()]
        delta = parameters['delta']
        if G is not None: E = G.ecount()
        else:
            avg_deg = parameters['avg_deg']
            N = np.sum(block_sizes)
            E = N * avg_deg / 2
        K_real = delta / np.sum(delta) * 2 * E
        communities_names = list(parameters['communities'].keys())
        # Plot
        df_path = experiment_row['path_to_measure']
        options_metadata = experiment_row['parameters']
        communities_names_ordered = sorted(communities_names)
        label_order = {i: communities_names_ordered.index(communities_names[i]) for i in range(len(communities_names))}
        _, ax = plot_K_ensamble(df_path, K_real, options_metadata, ax = axs.flat[i_plot], label_order = label_order, text_fontsize = 18)
        i_plot = i_plot + 1

        ax.set_xticks(np.arange(len(communities_names_ordered)), labels=communities_names_ordered, rotation = 45, ha = 'right')
        ax.set_yticks(np.arange(len(communities_names_ordered)), labels=communities_names_ordered)
    
    # Deactivate unused axes
    for ax in axs.flat[i_plot:]:
        ax.axis('off')
    
    # Set cbar
    im = axs.flat[0].get_images()[0]
    fig.colorbar(im, ax=[ax], orientation='vertical', shrink=0.8)
    cbar = fig.axes[-1]
    cbar.set_label('Relative error')

    fig.suptitle('K matrix errors')
    image_path = os.path.join(output_directory, file_name)
    fig.savefig(image_path)
    if verbose:
        print(f"K plot saved to: {image_path}")

def plot_degree(G, simulation_df, output_directory, file_name = 'degree.pdf', g_bm = None, benchmark_label_string = None, verbose = False):
    fig, ax = plt.subplots(figsize = (14, 9))
    bins = np.arange(1, 300, 1)
    if G is not None: plot_degree_from_graph(G, ax = ax, bins = bins)
    if g_bm is not None: plot_degree_from_graph(g_bm, ax = ax, bins = bins, label_string=benchmark_label_string)
    for id, experiment_row in simulation_df[simulation_df['measure'] == 'Degree'].iterrows():
        plot_degree_from_ensambles_distribution(experiment_row['path_to_measure'],
                                                options = experiment_row['parameters'],
                                                ax = ax, bins = bins, fit = False)
    image_path = os.path.join(output_directory, 'degree.pdf')
    ax.grid(True, which = 'major')
    ax.set_ylim((1e-4, 3))
    ax.set_xlim((1,300))
    ax.plot([],[], label = r'$\mathbb{E}[k]$', color = 'black', linestyle = (0, (5, 5)))
    ax.legend(loc = 'upper left', bbox_to_anchor = (1, 1.05))
    fig.tight_layout()
    fig.savefig(image_path, dpi = 300)
    if verbose:
        print(f"Degree plot saved to: {image_path}")

def plot_clustering(G, simulation_df, output_directory, file_name = 'clustering.pdf', g_bm = None, benchmark_label_string = None, verbose = False):
    fig, ax = plt.subplots(figsize = (14, 9))
    if G is not None: plot_clustering_from_graph(G, ax = ax)
    if g_bm is not None: plot_clustering_from_graph(g_bm, ax = ax, label = benchmark_label_string)
    for id, experiment_row in simulation_df[simulation_df['measure'] == 'Samplings'].iterrows():
        df_path = experiment_row['path_to_measure']
        options = experiment_row['parameters']
        plot_clustering_from_ensambles_distribution(df_path, options = options, ax = ax)

    image_path = os.path.join(output_directory, file_name)
    ax.set_ylim((0,1))
    ax.set_xlim((2,None))
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\bar{c}(k)$')
    ax.set_title(f'Clustering coefficient')
    ax.grid(True, which = 'major')
    ax.legend(loc = 'upper left', bbox_to_anchor = (1, 1.05))
    ax.xaxis.set_major_locator(ticker.FixedLocator([2, 10, 100, 1000]))
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = '$\\mathdefault{2\\times10^{0}}$'
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = ''
    ax.set_yticklabels(labels)
    fig.tight_layout()
    fig.savefig(image_path, dpi = 300)
    if verbose:
        print(f"Clustering plot saved to: {image_path}")

def plot_triangles(G, simulation_df, output_directory, file_name = 'triangles.png', verbose = False):
    triangles = get_triangles_from_G(G)
    fig, ax = plt.subplots(figsize = (14, 9))
    # discard first color
    ax.set_prop_cycle(color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
    scatter_data = []
    for id, experiment_row in simulation_df[simulation_df['measure'] == 'Triangles'].iterrows():
        df_path = experiment_row['path_to_measure']
        options = experiment_row['parameters']
        scatter_data.append(plot_triangle_data(df_path, triangles, options = options, ax = ax))

    image_path = os.path.join(output_directory, file_name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim((1e-2, None))
    ax.set_xlabel('original number of triangles')
    ax.set_ylabel('inferred number of triangles')
    ax.plot([1, 2*G.vcount()], [1, 2*G.vcount()], linestyle = (0, (10, 10)), color = '#000000')
    ax.set_title(f'Triangles')
    ax.set_xlim((1, 2*G.vcount()))
    ax.set_ylim((None, 2*G.vcount()))
    ax.grid(True, which = 'major')
    handles, labels = ax.get_legend_handles_labels()
    legend_handles = [copy.copy(h) for h in handles]
    for h in legend_handles:
        h.set_alpha(1)
    ax.legend(legend_handles, labels, loc = 'upper left', bbox_to_anchor = (1, 1.05))
    fig.tight_layout()
    fig.savefig(image_path, dpi = 300)
    if verbose:
        print(f"Triangles plot saved to: {image_path}")
    
def plot_annd(G, simulation_df, output_directory, file_name = 'annd.png', verbose = False):
    fig, ax = plt.subplots(figsize = (14, 9))
    scatter_data = []
    original_annd = G.knn()[0]
    # discard first color
    ax.set_prop_cycle(color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
    for id, experiment_row in simulation_df[simulation_df['measure'] == 'Average Neighbor Degree'].iterrows():
        df_path = experiment_row['path_to_measure']
        options = experiment_row['parameters']
        scatter_data.append(plot_annd_data(df_path, original_annd, options = options, ax = ax))
        
    image_path = os.path.join(output_directory, file_name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((1, G.vcount()))
    ax.set_ylim((1, G.vcount()))
    ax.set_xlabel('original average degree of neighbors')
    ax.set_ylabel('inferred average degree of neighbors')
    ax.plot([1, G.vcount()], [1, G.vcount()], linestyle = (0, (10, 10)), color = 'black')
    ax.set_title(f'Average Degree of Neighbors')
    ax.grid(True, which = 'major')
    handles, labels = ax.get_legend_handles_labels()
    legend_handles = [copy.copy(h) for h in handles]
    for h in legend_handles:
        h.set_alpha(1)
    ax.legend(legend_handles, labels, loc = 'upper left', bbox_to_anchor = (1, 1.05))
    fig.tight_layout()
    fig.savefig(image_path, dpi = 300)
    if verbose:
        print(f"Average Neighbor Degree plot saved to: {image_path}")

def plot_knn(G, simulation_df, output_directory, file_name = 'knn.pdf', g_bm = None, benchmark_label_string = None, verbose = False):
    fig, ax = plt.subplots(figsize = (14, 9))
    if G is not None: plot_knn_from_graph(G, ax = ax)
    if g_bm is not None: plot_knn_from_graph(g_bm, ax = ax, label = benchmark_label_string)

    for id, experiment_row in simulation_df[simulation_df['measure'] == 'Average Neighbor Degree'].iterrows():
        df_path = experiment_row['path_to_measure']
        options = experiment_row['parameters']
        data = plot_knn_from_ensambles_distribution(df_path, options = options, ax = ax)
        
    image_path = os.path.join(output_directory, file_name)
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\bar{k}_{nn}(k)$')
    ax.set_title(f'Average Neighbor Degree given the degree of the node')
    ax.grid(True, which = 'major')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(image_path, dpi = 300)
    if verbose:
        print(f"Average Neighbor Degree given the degree of the node plot saved to: {image_path}")            

def plot_global_metrics(G, simulation_df, output_directory, g_bm = None, file_name = 'global_metrics.tex', verbose = False):
    rows = []
    for id, experiment_row in simulation_df[simulation_df['measure'] == 'Samplings'].iterrows():
        df_path = experiment_row['path_to_measure']
        options = experiment_row['parameters']
        row = global_metrics_table_ensamble(df_path, options)
        rows.append(row)

    if G is not None:
        row = global_metrics_table_graph(G)
        rows.append(row)
    if g_bm is not None:
        row = global_metrics_table_graph(g_bm, index = 'SBM')
        rows.append(row)
    df_global = pd.concat([pd.DataFrame([row]) for row in rows])
    table_path = os.path.join(output_directory, file_name)
    make_latex_table(df_global, table_path)
    if verbose:
        print(f"Global metrics table saved to: {table_path}")
