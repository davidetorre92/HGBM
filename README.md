# Random Hyperbolic Graphs with Arbitrary Mesoscale Structures

The scripts in this repository allow the generation of random hyperbolic graphs with arbitrary mesoscale structures. The project is organized into two main modules, one handling network generation, and another for data organization and measurement analysis.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
  - [Run Simulations](#run-simulations)
  - [Utilities](#utilities)
  - [Measurement Configuration](#measurement-configuration)
- [License](#license)

## Installation

To download this repository run:

```bash
git clone https://github.com/davidetorre92/hgbm
```

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```
## Quickstart
To generate a random hyperbolic graph with 4 assortative communties, 500 vertices, powerlaw degree distribution with exponent $\alpha = 2.5$ and average degree of 10, inverse temperature $\beta = 10$ and 10 simulations, run the following command:

```bash
python hgbm/hgbm.py -c hgbm/config.py
```
To measure the properties of the generated network, run:

```bash
python measurements/measure.py -c measurements/config.py
```

Finally, to plot the results, run:

```bash
python measurements/plot.py -c measurements/config.py
```

## Modules

### Network Generation

#### Configuration file
```hgbm/config.py``` contains configuration settings for the simulation. Here's the available options that must be filled in a dictionary:
```
    "communities": {"0": [n for n in range(0,125)], 
                    "1": [n for n in range(125,250)],
                    "2": [n for n in range(250,375)],
                    "3": [n for n in range(375,500)]},
    "delta": [[4,1,1,1],[1,4,1,1],[1,1,4,1],[1,1,1,4]],
    "degrees": None,
    "beta": 10.0,
    "alpha": 2.5,
    "avg_deg": 10.0,
    "xmin": None,
    "verbose": True,
    "output_directory": "/home/davide/AI/Projects/HGBM/article_results/test_git/",
    "save_timestamp": False,
    # "graph_path": "/home/davide/AI/Projects/HGBM/graphs/grafo_elezioni.pickle",
    # "community_attribute": 'community',
    "n_tests": 10,
    "adjust_hidden_degrees": True
```
- ```communities```: A dictionary where the keys represent community labels and the values are lists of node indices that belong to that community.
- ```delta```: A square matrix representing the mixing matrix of the hyperbolic graph, which MUST be symmetrical.
- ```beta```: The inverse temperature parameter of the hyperbolic graph.
- ```alpha```: The exponent parameter of the power-law degree distribution.
- ```avg_deg```: The average degree of the hyperbolic graph.
- ```xmin```: The minimum value of the power-law degree distribution.

If the users wants to generate a graph with a given degree sequence, they can specify:
- ```degrees```: A list of degrees for each node.

If the user wants to randomize a graph where the vertices have an attribute indicating their communities, they can specify:
- ```graph_path```: The path to the graph file.
- ```community_attribute```: The name of the attribute that indicates the community.

The user cannot set ```avg_deg``` and ```alpha```, and ```graph_path``` and ```community_attribute``` at the same time.
#### Main script
- ```hgbm/hgbm.py```: This script manages the simulation process, including parameter handling and logging. It supports different modes for handling existing results. 

### Utilities

- **hgbm/utils.py**: Contains utility functions for network models, including power-law sampling and expected degree calculations.
- **measurements/utils_plot.py**: Provides functions for visualizing simulation results, such as plotting ensemble matrices and global metrics.

### Measurement Configuration

- ```measurements/config.py```: This directory includes various configuration scripts that define settings for different simulation scenarios and measurement collections.

## Batch simulations

The main entry point for running simulations is the `run_simulations.py` script. It allows users to specify various parameters for simulations and manage output logs effectively.

```bash
python run_simulations.py [options]
```
## License

This project is licensed under the MIT License - see the LICENSE file for details.
