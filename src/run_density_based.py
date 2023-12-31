from distances import DistanceMatrices
from itertools import product
from cluster.mountain import MountainClustering
import numpy as np
from cluster.validation import Validation
import pandas as pd



def get_params_mountain_algorithms(tols, sigmas, ras, betas = None, rbs=None  ):
    """Returns the parameters of the mountain algorithms
    Parameters:
        tols (list):
            List of tolerances
        sigmas (list):
            List of sigmas
        ras (list):
            List of ra
        betas (list):
            List of betas
        rbs (list):
            List of rb
    Returns:
        mountain_algo_params (dict):
            Dictionary with the parameters of the mountain algorithms
    """
    mountain_params = []
    if betas is not None:
        combinations_mountain = list(product(sigmas, betas, tols))
        for sigma, beta, tol in combinations_mountain:
            clustering_args = {
                'sigma' : sigma,
                'beta' : beta,
                'tol' : tol
            }
            mountain_params.append(clustering_args)
    else:
        comb = list(product(sigmas, tols))
        for sigma, tol in comb:
            clustering_args = {
                'sigma' : sigma,
                'tol' : tol
            }
            mountain_params.append(clustering_args)

    subtractive_mountain_params = []
    if rbs is not None:
        combinations_subtractive = list(product(ras, rbs, tols))
        for ra, rb, tol in combinations_subtractive:
            clustering_args = {
                'ra' : ra,
                'rb' : rb,
                'tol' : tol
            }
            subtractive_mountain_params.append(clustering_args)
    else:
        comb = list(product(ras, tols))
        for ra, tol in comb:
            clustering_args = {
                'ra' : ra,
                'tol' : tol
            }
            subtractive_mountain_params.append(clustering_args)
    
    mountain_algo_params = {
        'mountain': mountain_params,
        'subtractive':subtractive_mountain_params
    }
    
    return mountain_algo_params


def get_vertices_grid(len_grid,M):
    """Returns the vertices of a grid of M dimensions
    Parameters:
        len_grid (int):
            Number of points in each dimension
        M (int):
            Number of dimensions
    Returns:
        vertices (numpy.ndarray):
            Vertices of the grid
    """
    alpha = 1/(len_grid-1)
    grid = [float(i*alpha) for i in range(len_grid)]
    lists_of_ranges = [grid for i in range(M)]
    tuple_ranges = tuple(lists_of_ranges)
    vertices = list(product(*tuple_ranges))
    vertices = np.array(vertices)
    return vertices

def run_mountain_algorithms(X, Y, dm, distance_definitions, grid_points, tols, sigmas, ras, betas = None, rbs=None, main_path = 'Iris', include_external = True):
    """Runs the mountain clustering algorithms
    Parameters:
        X (numpy.ndarray):
            Data to cluster
        Y (numpy.ndarray):
            Labels of the data
        dm (DistanceMatrices):
            DistanceMatrices object
        distance_definitions (dict):
            Dictionary with the distance definitions
        grid_points (list):
            List of number of grid points to try
        tols (list):
            List of tolerances
        sigmas (list):
            List of sigmas
        ras (list):
            List of ra
        betas (list):
            List of betas
        rbs (list):
            List of rb
        main_path (str):
            Path where the results will be saved
        include_external (bool):
            Whether to include external indexes or not
    Returns:
        df_indices (pandas.DataFrame):
            Dataframe with the internal and external indexes of the clustering algorithms
    """
    mountain_algos_params = get_params_mountain_algorithms(tols, sigmas, ras, betas = betas, rbs=rbs)


    dist_dictionary = dm.distance_matrices

    grids = {}

    for gp in grid_points:
        grid = get_vertices_grid(gp,X.shape[1])
        grids[gp] = grid

    indexes_results = []

    for dist_id in dist_dictionary:

        dist_mat = dist_dictionary[dist_id]

        dist_args = distance_definitions[dist_id]
        dist_name = dist_args['distance_name']

        for algo_name in mountain_algos_params:
            for clustering_args in mountain_algos_params[algo_name]:
                if algo_name == 'subtractive':
                    mc = MountainClustering(X, algo_name, clustering_args, dist_args, dist_name, distance_mat = dist_mat, main_path = main_path)
                    mc.cluster()
                    mc.save_results()
                    print(algo_name, clustering_args, dist_name, len(mc.centers))

                if algo_name == 'mountain':
                    for gp in grid_points:
    
                        grid = grids[gp]
                        distance_mat_v = dm.compute_distance_matrix_fast(X, grid, **dist_args)
                        mc = MountainClustering(X, algo_name, clustering_args, dist_args, dist_name, distance_mat_v = distance_mat_v, grid=grid, grid_points = gp, main_path = main_path)
                        mc.cluster()
                        mc.save_results()
                        print(algo_name, clustering_args, dist_name, gp, len(mc.centers))
                kc_val = Validation(mc.memberships, X, dist_args, centroids = mc.centers)
                internal_indexes = kc_val.get_all_internal_indexes()
                if include_external:
                    external_indexes = kc_val.get_all_external_indexes(Y)
                    indexes_results.append((algo_name, mc.params_strs, dist_name, *internal_indexes.values() , *external_indexes.values(), len(mc.centers)))
                else:
                    indexes_results.append((algo_name, mc.params_strs, dist_name, *internal_indexes.values() , len(mc.centers)))
    
    if include_external:  
        df_indices = pd.DataFrame(indexes_results, columns= ['Algorithm', 'Parameters', 'Distance', 'CH', 'BH', 'Hartigan', 'xu', 'DB', 'S', 'Rand', 'Fowlkes-Mallows', 'Jaccard', 'Number of clusters'])
    else:
        df_indices = pd.DataFrame(indexes_results, columns= ['Algorithm', 'Parameters', 'Distance', 'CH', 'BH', 'Hartigan', 'xu', 'DB', 'S', 'Number of clusters'])
    return df_indices





    
    