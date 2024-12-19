import numpy as np

config = {
    "communities": {"0": [n for n in range(0,1250)], 
                    "1": [n for n in range(1250,2500)],
                    "2": [n for n in range(2500,3750)],
                    "3": [n for n in range(3750,5000)]},
    "delta": [[1.2,1,1,1],[1,1.2,1,1],[1,1,1.2,1],[1,1,1,1.2]],
    "degrees": None,
    "beta": 2.0,
    # "alpha": 2.5,
    # "avg_deg": 10.0,
    "xmin": None,
    "verbose": True,
    "output_directory": "/home/davide/AI/Projects/HGBM/article_data/elezioni/beta_2/",
    "save_timestamp": False,
    "graph_path": "/home/davide/AI/Projects/HGBM/graphs/grafo_elezioni.pickle",
    "community_attribute": 'community',
    "n_tests": 10,
    "adjust_hidden_degrees": True
}
