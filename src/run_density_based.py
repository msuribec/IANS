from distances import DistanceMatrices
from itertools import product
from cluster.mountain import MountainClustering
import numpy as np
from sklearn import datasets
from cluster.validation import Validation
import pandas as pd



def get_params_mountain_algorithms(tols, sigmas, ras, betas = None, rbs=None  ):

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
        combinations_subtractive = list(product(ras, rbs, tols)) #TODO:_ remove duplicates
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
    alpha = 1/(len_grid-1)
    grid = [float(i*alpha) for i in range(len_grid)]
    lists_of_ranges = [grid for i in range(M)]
    tuple_ranges = tuple(lists_of_ranges)
    vertices = list(product(*tuple_ranges))
    vertices = np.array(vertices)
    return vertices

def run_mountain_algorithms(X, Y, distance_definitions, grid_points, tols, sigmas, ras, betas = None, rbs=None, main_path = 'Iris'):

    #TODO: guardar en un dataframe los resultados de cada algoritmo
    # coger self.params_str
    mountain_algos_params = get_params_mountain_algorithms(tols, sigmas, ras, betas = betas, rbs=rbs)


    dm = DistanceMatrices(main_path)

    dm.compute_distance_matrices(X, distance_definitions)
    dist_dictionary = dm.distance_matrices

    grids = {}

    for gp in grid_points:
        grid = get_vertices_grid(gp,X.shape[1])
        grids[gp] = grid

    indexes_results = []
    internal_indexes_results = []
    external_indexes_results = []
    number_clusters = []

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
                        #TODO SAVE RESULTS
                        #TODO GRAPH RESULTS
                kc_val = Validation(mc.memberships, X, dist_args, centroids = mc.centers)

                internal_indexes = kc_val.get_all_internal_indexes()
                external_indexes = kc_val.get_all_external_indexes(Y)


                indexes_results.append((algo_name, mc.params_strs, dist_name, *internal_indexes.values() , *external_indexes.values(), len(mc.centers)))
                internal_indexes_results.append((algo_name, mc.params_strs, dist_name, *internal_indexes.values()))
                external_indexes_results.append((algo_name, mc.params_strs, dist_name, *external_indexes.values()))
                number_clusters.append((algo_name, mc.params_strs, dist_name, len(mc.centers)))

    df_indices = pd.DataFrame(indexes_results, columns= ['Algorithm', 'Parameters', 'Distance', 'CH', 'BH', 'Hartigan', 'xu', 'DB', 'S', 'Rand', 'Fowlkes-Mallows', 'Jaccard', 'Number of clusters'])
    df_internal_indices = pd.DataFrame(internal_indexes_results, columns= ['Algorithm', 'Parameters', 'Distance', 'CH', 'BH', 'Hartigan', 'xu', 'DB', 'S'])
    df_external_indices = pd.DataFrame(external_indexes_results, columns= ['Algorithm', 'Parameters', 'Distance', 'Rand', 'Fowlkes-Mallows', 'Jaccard'])
    df_nclusters = pd.DataFrame(number_clusters, columns= ['Algorithm', 'Parameters', 'Distance', 'Number of clusters'])

    return df_indices





    
    
