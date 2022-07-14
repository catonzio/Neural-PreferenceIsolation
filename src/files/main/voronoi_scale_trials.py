import numpy as np
import time
from files.pif.voronoi_iforest import *

ivor_parameters: dict = {'num_trees': 100, 'max_samples': 256, 'branching_factor': 2, 'metric': 'tanimoto',
                         'n_jobs': -1}

for cols in [10, 100, 500, 1000, 10000]:
    st = time.time()
    preference_matrix = np.random.uniform(size=(10000, cols))
    voronoi = VoronoiIForest(**ivor_parameters)
    voronoi.fit(preference_matrix)
    print(f"With {preference_matrix.shape}, time of execution:\t{time.time() - st}")