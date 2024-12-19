import numpy as np

config = {
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
}
