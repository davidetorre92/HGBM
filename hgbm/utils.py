import numpy as np
import powerlaw
from datetime import datetime
import importlib.util
import os
import igraph as ig
from collections import Counter
from scipy.special import hyp2f1

import pdb

BETA_DEFAULT = 10
ALPHA_DEFAULT = 2.5
AHD_DEFAULT = False
PROBABILITY_FILENAME_DEFAULT = "hgbm.npy"
THETA_FILENAME_DEFAULT = "theta.npy"
FITNESS_FILENAME_DEFAULT = "fitness.npy"
INPUT_FILENAME_DEFAULT = "input.npy"
FOLDER_DEFAULT = "./results/hgbm_default"
N_TEST_DEFAULT = 10

def create_folder(path, verbose):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        if verbose: print_time(f"Output directory created: {path}")
    else:
        if verbose: print_time(f"Output directory {path} already exists")
    return None
def add_test_id(filename, test_id = None):
    if test_id is None:
        return filename
    else:
        basename = os.path.basename(filename)
        extention = os.path.splitext(basename)[1]
        return f"{basename.split('.')[0]}_{test_id}{extention}"
def print_time(string, begin = None, end = '\n'):
    if begin is None:
        time = datetime.now()
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {string}", end = end)
    else:
        print(f"{begin}{string}")

def save_triu(path, A):
    # Get the indices of the upper triangle
    triu_indices = np.triu_indices_from(A)

    # Extract the upper triangle values
    upper_triangle_values = A[triu_indices]

    # Save the upper triangle values
    np.save(path, upper_triangle_values)
    print_time(f"Proability matrix saved in {path}")

def load_triu(path):
    # Load the upper triangle values
    upper_triangle_values = np.load(path)

    # Reconstruct the original matrix
    N = int(np.sqrt(upper_triangle_values.size * 2))  # Determine the size of the original matrix
    reconstructed_matrix = np.zeros((N, N))

    # Get the indices of the upper triangle
    triu_indices = np.triu_indices(N)

    # Place the values back into the reconstructed matrix
    reconstructed_matrix[triu_indices] = upper_triangle_values

    # Since the matrix is symmetrical, copy the upper triangle to the lower triangle
    reconstructed_matrix = reconstructed_matrix + reconstructed_matrix.T - np.diag(np.diag(reconstructed_matrix))
    return reconstructed_matrix

def contains_subdirectory(directory):
    for _, dirs, _ in os.walk(directory):
        if dirs:
            for dir in dirs:
                if dir not in ['.', '..', 'measurements']:
                    return False
            return True
    return False

def load_config_file(path, mode = 'experiments'):
    config = importlib.util.spec_from_file_location("config", path).loader.load_module()
    input_data = config.config
    input_data_formatted = check_and_format_configuration(input_data, mode)

    return input_data_formatted

def check_data_concistencies(communities, delta, verbose = True):
  if delta.shape[0] != delta.shape[1]:
    raise ValueError('Mixing matrix delta must be a square matrix.')
  if np.any(delta.T != delta):
    raise ValueError('Mixing matrix delta must be symmetrical.')

  if type(communities) == dict:
    if verbose: print("Communities are dictionaries: each key contains the indices of the nodes in that community")
    n_communities = len(communities.keys())
    if verbose: print(n_communities, "communities found")
  elif type(communities) == np.ndarray:
    if verbose: print("Communities are lists: each element contains the total number of nodes in that community")
    n_communities = communities.shape[0]
  else:
      ValueError("communities {type(communities)} not implemented. Aborting")
  if n_communities != delta.shape[0]:
    raise ValueError('Inconcistency between the number of communities in n_communities and delta.')
  


def get_N(communities):
    return np.sum([np.sum(len(comm)) for comm in communities.values()])
def get_E(N, avg_deg):
    return N * avg_deg / 2
def get_K(delta, E):
    proportions = delta / np.sum(delta)
    print("Normalized Delta:")
    print(proportions * 2)
    K = proportions * 2 * E
    return K

def expected_measures(communities, delta, avg_deg):

    N = get_N(communities)
    E = get_E(N, avg_deg)
    K = get_K(delta, E)

    return N, E, K

def get_node_index_to_community(N, communities, communities_names):
    community_to_integer = {comm: i for i, comm in enumerate(communities_names)}
    node_to_communities_int = np.zeros(N, dtype = int)
    for community, indices in communities.items():
        for index in indices:
            node_to_communities_int[index] = community_to_integer[community]
    node_to_communities_int = np.array(node_to_communities_int, dtype = int)
    return node_to_communities_int

def get_communities_from_attribute(G, attribute = None):

    if bool(attribute):
        # Check if attribute exist:
        if attribute not in G.vs.attributes():
            raise ValueError(f"Attribute {attribute} not found in graph. Aborting")
        communities_names = list(set(G.vs[attribute]))
        n_communities = len(communities_names)
        communities = {name: [] for name in communities_names}
        community_to_int = {comm: i for i, comm in enumerate(communities_names)}

        node_to_communities = {}
        node_to_communities_int = np.zeros(G.vcount(), dtype = int)

        for v in G.vs:
            index = v.index
            node_community = v[attribute]
            attribute_index = community_to_int[node_community]

            communities[node_community].append(index)
            node_to_communities[index] = node_community
            node_to_communities_int[index] = attribute_index

        edges_between_comminities_counter = np.zeros((n_communities,n_communities))
        is_dir = G.is_directed()
        for e in G.es:
            s = e.source
            t = e.target
            comm_s = node_to_communities[s]
            comm_t = node_to_communities[t]
            i_comm_s = community_to_int[comm_s]
            i_comm_t = community_to_int[comm_t]
            edges_between_comminities_counter[i_comm_s, i_comm_t] += 1
            if is_dir is False: edges_between_comminities_counter[i_comm_t, i_comm_s] += 1

        delta = edges_between_comminities_counter / G.ecount()
    else:
        delta = np.array([[2.]])
        communities_names = ['0']
        communities = {'0': [i for i in range(G.vcount())]}
    return delta, communities_names, communities
#############
#
# Samplings
#
#############

def sample_angle(N, thetas):
    if thetas is not None:
        return thetas
    else:
        return np.random.uniform(0, 2 * np.pi, N)

def find_xmin_from_expected_value(exp_value, alpha):
    return (alpha - 2) / (alpha - 1) * exp_value

def sample_latent_coordinate_powerlaw(N, avg_deg, alpha, xmin = None):
  """
  Sampling the latent coordinate with power law distribution.
  """
  if xmin is None: xmin = find_xmin_from_expected_value(avg_deg, alpha)
  theoretical_distribution = powerlaw.Power_Law(xmin=xmin, parameters=[alpha])
  return theoretical_distribution.generate_random(N)

def get_R(N):
  return N / (2 * np.pi)

def delta_theta_ij_vectorized(R, theta):
    theta_i = theta[:, np.newaxis]
    theta_j = theta[np.newaxis, :]
    x_ij = (np.pi - np.abs(np.pi - np.abs(theta_i - theta_j))) * R
    return x_ij

def get_p_ij_vectorized(fitness, theta, beta, N, K, sum_block_fitness, node_index_to_block):

    # Expanded to all pairs
    I = node_index_to_block[:, np.newaxis] # column vec
    J = node_index_to_block[np.newaxis, :] # row vec

    # Collect sums for each block
    sum_f_u_j = sum_block_fitness[J]
    sum_f_u_i = sum_block_fitness[I]

    K_IJ = K[I, J]
    R = get_R(N)
    x_ij = delta_theta_ij_vectorized(R, theta)

    f_i = fitness[:, np.newaxis]  # Convert to column vector
    f_j = fitness[np.newaxis, :]  # Convert to row vector

    # Calculate denominator terms    
    den_den = (f_i * f_j) / (sum_f_u_i * sum_f_u_j) * beta * (np.sin(np.pi / beta) / (2 * np.pi)) * N * K_IJ
    # Safely compute t to avoid division by zero
    t = np.where(den_den != 0, x_ij / den_den, np.inf)

    # Calculate p_ij, ensuring no invalid values are generated
    p_ij = np.where(np.isfinite(t), 1 / (1 + np.power(t, beta)), 0)

    # t = x_ij / den_den

    # p_ij = 1 / (1 + np.power(t, beta))
    # np.fill_diagonal(p_ij, 0)
    return p_ij

def get_fitness(N, alpha, avg_deg, degrees, xmin, 
                adjust_hidden_degrees, beta, K, node_to_communities_int, 
                verbose = False):
    if degrees is not None:
       return degrees
    else:
        if adjust_hidden_degrees:
            fitness = sample_latent_coordinate_powerlaw(N, avg_deg, alpha, xmin)
            fitness_adjusted = get_adjusted_hidden_degrees(fitness, beta, K, node_to_communities_int, verbose)
            return fitness_adjusted
        else:
            return sample_latent_coordinate_powerlaw(N, avg_deg, alpha, xmin)
def get_2f1(x, beta):
    return hyp2f1(1, 1/beta, 1+1/beta, -(x**beta))

def I_beta(beta):
    if beta == np.inf:
        return 0
    return np.pi/(beta*np.sin(np.pi/beta))

def adjust_degrees(input_data):
    beta = input_data['beta']
    fitness = input_data['degrees']
    communities = input_data['communities']
    communities_names = input_data['communities_names']
    delta = input_data['delta']
    avg_deg = input_data['avg_deg']
    N, _, K = expected_measures(communities, delta, avg_deg)
    node_to_communities_int = get_node_index_to_community(N, communities, communities_names)
    verbose = input_data['verbose']
    fitness_adjusted = get_adjusted_hidden_degrees(fitness, beta, K, node_to_communities_int, verbose)
    return fitness_adjusted

def expected_degree_per_k_blocks(hidden_degrees, beta, K, node_to_communities_int, round_degs=True):
    if round_degs:
        hidden_degrees = np.round(hidden_degrees)
    else:
        hidden_degrees = hidden_degrees.copy()
    block_deg_count = {}
    H = np.zeros(K.shape)
    for I in range(len(K)):
        this_degs = hidden_degrees[node_to_communities_int==I]
        block_deg_count[I] = Counter(this_degs)
        for J in range(len(K)):
            H[I,J] = K[I].sum()*K[J].sum()/K[I,J]
    exp_degs_per_k_block = np.zeros(len(hidden_degrees))      
    for I in range(len(K)):
        for k1 in block_deg_count[I].keys():
            x = 0
            for J in range(len(K)):
                x += sum(count*get_2f1(H[I,J]*I_beta(beta)/(k1*k2),beta) for k2,count in block_deg_count[J].items())
            exp_degs_per_k_block[(node_to_communities_int==I)&(hidden_degrees==k1)] = x
    return exp_degs_per_k_block.round()

def get_adjusted_hidden_degrees(real_degrees, beta, K, node_to_communities_int, verbose, err=1, max_iter=100, patience = 10):
    N = len(real_degrees)
    keep_going = True
    cnt = 0
    hidden_degrees = real_degrees.copy()
    patience_cnt = 0
    n_changes = -1
    while keep_going and (cnt < max_iter):
        if verbose: print_time(f"Adjusting hidden degrees: iteration {cnt}.\t\t", end='\r')
        exp_deg_per_k_block = expected_degree_per_k_blocks(hidden_degrees, beta, K, node_to_communities_int)
                
        keep_going = False
        if np.any(np.abs(exp_deg_per_k_block-real_degrees)>err):
            if patience_cnt == 0:
                # Count how many nodes have changed their degree
                n_changes_last = np.sum(np.abs(exp_deg_per_k_block-real_degrees)>err)
                patience_cnt = 1
            else:
                n_changes = np.sum(np.abs(exp_deg_per_k_block-real_degrees)>err)
                if n_changes == n_changes_last:
                    patience_cnt += 1
                else:
                    patience_cnt = 0
                    n_changes_last = n_changes
            if patience_cnt > patience:
                keep_going = False
            else:
                keep_going = True
            hidden_degrees = hidden_degrees + (real_degrees-exp_deg_per_k_block)*np.random.random(N)
        cnt += 1
    return hidden_degrees    
#############
#
# Graph definition
#
#############

def check_and_format_configuration(input_data, mode):
    if type(input_data) != dict:
        raise ValueError("Input data must be a dictionary")

    communities             = input_data.get('communities')
    communities_names       = input_data.get('communities_names')
    delta                   = input_data.get('delta')
    alpha                   = input_data.get('alpha')
    xmin                    = input_data.get('xmin')
    avg_deg                 = input_data.get('avg_deg')
    beta                    = input_data.get('beta')
    degrees                 = input_data.get('degrees')
    thetas                  = input_data.get('thetas')
    graph_path              = input_data.get('graph_path') 
    community_attribute     = input_data.get('community_attribute')
    adjust_hidden_degrees   = input_data.get('adjust_hidden_degrees') 
    verbose                 = input_data.get('verbose')
    n_tests                 = input_data.get('n_tests')
    input_filename          = input_data.get('input_filename')
    fitness_filename        = input_data.get('fitness_filename')
    theta_filename          = input_data.get('theta_filename')
    probability_filename    = input_data.get('probability_filename')
    output_directory        = input_data.get('output_directory')
    save_timestamp          = input_data.get('save_timestamp')

    # Set default values
    if bool(verbose):
        print_time("Verbosity activated")
    else:
        input_data['verbose'] = False
    # Print beginning message and parameters
    if verbose:
        print("----------"*5)
        print("Hyperbolic geometric block model")
        print("----------"*5)
        print("Parameters:")
        print(f"beta = {input_data['beta']}", end = '\n\n\n')

    if type(input_filename) is not str:
        if verbose: print_time(f"Warning fitness_filename not set, using default value {INPUT_FILENAME_DEFAULT}")
        input_data['input_filename'] = INPUT_FILENAME_DEFAULT
    if type(fitness_filename) is not str:
        if verbose: print_time(f"Warning fitness_filename not set, using default value {FITNESS_FILENAME_DEFAULT}")
        input_data['fitness_filename'] = FITNESS_FILENAME_DEFAULT
    if type(theta_filename) is not str:
        if verbose: print_time(f"Warning theta_filename not set, using default value {THETA_FILENAME_DEFAULT}")
        input_data['theta_filename'] = THETA_FILENAME_DEFAULT
    if type(probability_filename) is not str:
        if verbose: print_time(f"Warning probability_filename not set, using default value {PROBABILITY_FILENAME_DEFAULT}")
        input_data['probability_filename'] = PROBABILITY_FILENAME_DEFAULT
    if type(output_directory) is not str:
        if verbose: print_time(f"Warning probability_filename not set, using default value {FOLDER_DEFAULT}")
        input_data['output_directory'] = FOLDER_DEFAULT
    if bool(save_timestamp):
        out_folder = datetime.now().strftime("%Y%m%d-%H%M%S")
        input_data['output_directory'] = os.path.join(input_data['output_directory'], out_folder)
    if n_tests is None:
        n_tests = N_TEST_DEFAULT
        if verbose: print_time(f"Warning n_tests not set, using default value {n_tests}")
    else:
        if type(n_tests) != int:
            n_tests = N_TEST_DEFAULT
            if verbose: print_time(f"Warning n_tests not an integer, using default value {n_tests}")
    input_data['n_tests'] = n_tests
    if type(beta) is not float:
        beta = BETA_DEFAULT
        if verbose: print_time(f"Warning beta not set, using default value {beta}")

# Concistency checks and data preparation
    if mode == 'experiments':
        # Check if either graph_path, avg_deg and alpha, or degrees are set
        read_mode = bool(graph_path) + bool(degrees) + (bool(avg_deg) and bool(alpha))
        # pdb.set_trace()
        if read_mode != 1:
            if verbose: print(f"graph_path: {graph_path}\ndegrees: {degrees}\navg_deg: {avg_deg}\nalpha: {alpha}")
            raise ValueError("Either graph_path or degrees, avg_deg and alpha must be set")
        # Beta must be set regardless of the read mode
        if bool(input_data['beta']) is False:
            beta = BETA_DEFAULT
            if verbose: print_time(f"Warning beta not set, using default value {beta}")
        else:
            beta = input_data['beta']
            
        if graph_path is None:
        # Check that the input data is in the right format
            if type(communities) is not dict:
                raise ValueError("Communities must be a dictionary")
            if type(communities_names) is not list:
                communities_names = list(communities.keys())
                if verbose: print_time(f"Unable to read communities names, using default names {communities_names}")
            if type(delta) is not list:
                raise ValueError("delta must be a list")
            N = get_N(communities)
            if bool(degrees) == (bool(avg_deg) and bool(alpha)):
                raise ValueError("Either degrees or avg_deg and alpha must be set")
            # Check on degrees
            if bool(degrees):
                if verbose: print_time("Reading degrees from a list")
                if type(degrees) == list or type(degrees) == np.ndarray:
                    if len(degrees) != N:
                        raise ValueError("degrees must be a vector of length N")
                    else:
                        degrees = np.array(degrees)
                        avg_deg = np.mean(degrees)
                        alpha = None
                elif type(degrees) == str:
                    if os.path.exists(degrees):
                        degrees = np.load(degrees)
                        avg_deg = np.mean(degrees)
                        alpha = None
                    else:
                        raise ValueError(f"Degrees file {degrees} not found")
                else:
                    raise ValueError("degrees must be a list or a numpy array")
            else:
                if verbose: print_time("Reading avg_deg and alpha to sample k from a powerlaw distribution") 
            # Check on avg_deg and alpha
                if type(alpha) is not float:
                    raise ValueError("alpha must be a float")
                if type(avg_deg) is not float:
                    raise ValueError("avg_deg must be a float")
            if (type(xmin) is float or xmin is None) is False:
                raise ValueError("xmin must be a float or None")                
            ## Convert to numpy for faster computation
            delta = np.array(delta)    
            check_data_concistencies(communities, delta)
        else:
            if verbose: print_time(f"Reading graph from {graph_path}")
            if type(graph_path) is not str:
                raise ValueError("graph_path must be a string")
            extension = os.path.splitext(graph_path)[1]
            if extension == '.graphml':
                G = ig.Graph.Read_GraphML(graph_path)
            elif extension == '.gml':
                G = ig.Graph.Read_GML(graph_path)
            elif extension == '.pickle':
                G = ig.Graph.Read_Pickle(graph_path)
            else:
                raise ValueError(f"Extension {extension} not supported")
            if verbose: print_time(f"Graph summary:\n\tpath: {graph_path}\n\t{G.summary()}")
            if community_attribute is not None:
                if verbose: print_time(f"Reading communities from attribute {community_attribute}")
            else:
                if verbose: print_time("community_attribute not set. Skipping communities reading")
            delta, communities_names, communities = get_communities_from_attribute(G, community_attribute)
            degrees = np.array(G.degree())
            avg_deg = np.mean(degrees)
            alpha = None
        # Set adjust_hidden_degrees flag
        if adjust_hidden_degrees is None:
            adjust_hidden_degrees = AHD_DEFAULT
            if verbose: print_time(f"Warning adjust_hidden_degrees not set, using default value {AHD_DEFAULT}")
        else:
            adjust_hidden_degrees = bool(adjust_hidden_degrees)
        
        input_data['beta'] = beta
        input_data['degrees'] = degrees
        input_data['avg_deg'] = avg_deg
        input_data['alpha'] = alpha
        input_data['xmin'] = xmin
        input_data['delta'] = delta
        input_data['communities_names'] = communities_names
        input_data['communities'] = communities
        input_data['adjust_hidden_degrees'] = adjust_hidden_degrees
        if adjust_hidden_degrees and degrees is not None:
        # Adjust degrees once for all when randomizing a network
            fitness_adjusted = adjust_degrees(input_data)
        else:
            fitness_adjusted = None
        input_data['degrees_adjusted'] = fitness_adjusted
        
    elif mode == 'measurements':
        pass
    else:
        raise ValueError("mode must be either 'experiments' or 'measurements'")
    if bool(thetas):
        if type(thetas) == list or type(thetas) == np.ndarray:
            if len(thetas) != N:
                raise ValueError("thetas must be a vector of length N")
            else:
                input_data['thetas'] = np.array(thetas)
        elif type(thetas) == str:
            if os.path.exists(thetas):
                input_data['thetas'] = np.load(thetas)
            else:
                raise ValueError(f"thetas file {thetas} not found")
    else:
        input_data['thetas'] = None
    
    return input_data

def get_community_from_index(index, communities):
    for community in communities.keys():
        if index in communities[community]:
            return community

def sample_graph_from_formatted_input(input_data):
    communities             = input_data['communities']
    communities_names       = input_data['communities_names']
    delta                   = input_data['delta']
    alpha                   = input_data['alpha']
    beta                    = input_data['beta']
    avg_deg                 = input_data['avg_deg']
    degrees                 = input_data['degrees']
    thetas                  = input_data['thetas']
    xmin                    = input_data['xmin']
    verbose                 = input_data['verbose']
    adjust_hidden_degrees   = input_data['adjust_hidden_degrees']
    if adjust_hidden_degrees and degrees is not None:
        degrees             = input_data['degrees_adjusted']
    
    N, E, K = expected_measures(communities, delta, avg_deg)
    n_communities = len(communities_names)
    node_to_communities_int = get_node_index_to_community(N, communities, communities_names)

    if verbose:
        print_time("Expected measures")
        print_time(f"N: {N}", begin = '\t')
        print_time(f"expected E: {E}", begin = '\t')
        print_time(f"K", begin = '\t')
        print_time(K, begin = '\t')

# Initialization
    fitness = get_fitness(N, alpha, avg_deg, degrees, xmin, 
                          adjust_hidden_degrees, beta, K, node_to_communities_int, 
                          verbose = verbose)
    theta = sample_angle(N, thetas)

# Get data from the blocks
    sum_block_fitness = np.zeros(n_communities)
    for i_N_b in range(len(communities_names)):
        sum_block_fitness[i_N_b] = np.sum(K[i_N_b])

    p_ij = get_p_ij_vectorized(fitness, theta, beta, N, K, sum_block_fitness, node_to_communities_int)
    # Set diagonal to zero to avoid self energies terms for each node.
    np.fill_diagonal(p_ij, 0)

    return fitness, theta, p_ij