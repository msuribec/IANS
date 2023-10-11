from distances import DistanceMatrices
from itertools import product
from cluster.mountain import MountainClustering
import numpy as np
from sklearn import datasets
from cluster.validation import Validation
import pandas as pd


def get_best_model(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df['punctuation'] = 0
    distances = np.unique(df['Distance'].values)
    algorithms = np.unique(df['Algorithm'].values)

    df_comparison = df
    indices = ['CH', 'BH',  'Hartigan',  'xu',  'DB',   'S',  'Rand','Fowlkes-Mallows','Jaccard']
    are_min =   [False, True,   False,     True,  True, False,  False ,  False,            False]
    for index, is_min in zip(indices,are_min):
    
        sorted_df = df_comparison.sort_values(by=[index],ascending=is_min)
        vals_criteria = sorted_df[index].values
        max_vals_criteria = np.max(vals_criteria)
        min_vals_criteria = np.min(vals_criteria)
        old_punctuation = sorted_df['punctuation'].values
        new_values = (vals_criteria - min_vals_criteria)/( max_vals_criteria- min_vals_criteria)
        sorted_df['punctuation'] = old_punctuation + new_values
        df_comparison = sorted_df
    sorted_df = df_comparison.sort_values(by=['punctuation'],ascending=[False])
    sorted_df['punctuation'] = sorted_df['punctuation']/len(indices)

    best_overall = (sorted_df.iloc[0]).to_dict()
    results_algos = {}
    results_algo_dist = {}

    distances = np.unique(df['Distance'].values)
    algorithms = np.unique(df['Algorithm'].values)

    for algo in algorithms: 
        dict_result_algo_dist = {}
        filter = sorted_df['Algorithm']  == algo
        algo_df = sorted_df.where(filter).dropna()
        results_algos[algo] = (algo_df.iloc[0]).to_dict()
        for distance in distances:
            filter2 = algo_df['Distance']  == distance
            algo_dist_df = algo_df.where(filter2).dropna()
            dict_result_algo_dist[distance] = (algo_dist_df.iloc[0]).to_dict()
        results_algo_dist[algo] = dict_result_algo_dist
    
    print("Best overall")
    print(best_overall)
    print("Best results by algorithm")
    print(results_algos)
    print("Best results by algorithm and distance")
    print(results_algo_dist)
    best_k = best_overall['Number of clusters']
    print(best_k)
    return best_overall, results_algos, results_algo_dist, best_k



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
    
    mountain_algos_params = get_params_mountain_algorithms(tols, sigmas, ras, betas = betas, rbs=rbs)


    dm = DistanceMatrices(main_path)

    dm.compute_distance_matrices(X, distance_definitions)
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
                    print(algo_name, clustering_args, dist_name)
                    print(len(mc.centers))

                if algo_name == 'mountain':
                    for gp in grid_points:
    
                        grid = grids[gp]
                        distance_mat_v = dm.compute_distance_matrix_fast(X, grid, **dist_args)
                        mc = MountainClustering(X, algo_name, clustering_args, dist_args, dist_name, distance_mat_v = distance_mat_v, grid=grid, grid_points = gp, main_path = main_path)
                        mc.cluster()
                        mc.save_results()
                        print(algo_name, clustering_args, dist_name, gp)
                        print(len(mc.centers))
                        #TODO GRAPH RESULTS
                kc_val = Validation(mc.memberships, X, dist_args, centroids = mc.centers)

                internal_indexes = kc_val.get_all_internal_indexes()
                external_indexes = kc_val.get_all_external_indexes(Y)
                indexes_results.append((algo_name, mc.params_strs, dist_name, *internal_indexes.values() , *external_indexes.values()))


    df_indices = pd.DataFrame(indexes_results, columns= ['Algorithm', 'Parameters', 'Distance', 'CH', 'BH', 'Hartigan', 'xu', 'DB', 'S', 'Rand', 'Fowlkes-Mallows', 'Jaccard'])
    best_overall, results_algos, results_algo_dist = get_best_model(df_indices)
    df_indices.to_csv(f'{main_path}/indices.csv')
    

def test_run(X, Y):

    inv_cov_matrix = np.linalg.inv(np.cov(X, rowvar=False))

    distance_definitions = {
        'Euclidean': {
            'distance_name': 'Euclidean'
        },
        'Manhattan':{
            'distance_name': 'Manhattan'
        },
        'Minkowski with p = 3': {
            'distance_name': 'Minkowski',
            'p' : 3
        },
        'Mahalanobis':{
            'distance_name': 'Mahalanobis',
            'inv_covmat': inv_cov_matrix
        },
        'Cosine':{
            'distance_name': 'Cosine',
        }
    }

    sigmas = [0.1,0.2,0.3,0.4,0.5]
    tols = [0.9,0.8,0.7,0.5,0.3]

    ras = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

    run_mountain_algorithms(X, Y, distance_definitions, [11], tols, sigmas, ras, betas = None, rbs=None, main_path = 'Iris')

    
