from run_naive import run_all_naive_algorithms
from run_density_based import run_mountain_algorithms
from runkmeans import run_kmeans_algorithms
from process import CleanData
from distances import DistanceMatrices
from utils import ReadData
from process import process_data
import sys
import numpy as np
from autoencoder import find_best_low_high_dimension_data
from embedding import UMAP_EMBEDDING
import ast


def get_best_model(df, path, main_path, include_external = True):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df['punctuation'] = 0
    distances = np.unique(df['Distance'].values)
    algorithms = np.unique(df['Algorithm'].values)

    df_comparison = df
    if include_external:
        indices = ['CH', 'BH',  'Hartigan',  'xu',  'DB',   'S',  'Rand','Fowlkes-Mallows','Jaccard']
        are_min =   [False, True,   False,     True,  True, False,  False ,  False,            False]
    else:
        indices = ['CH', 'BH',  'Hartigan',  'xu',  'DB',   'S']
        are_min =   [False, True,   False,     True,  True, False]        
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

    best_overall = sorted_df.iloc[0].to_dict()
    results_algos = {}
    results_algo_dist = {}

    distances = np.unique(df['Distance'].values)
    algorithms = np.unique(df['Algorithm'].values)

    for algo in algorithms: 
        dict_result_algo_dist = {}
        filter = sorted_df['Algorithm']  == algo
        algo_df = sorted_df.where(filter).dropna()
        results_algos[algo] = algo_df.iloc[0].to_dict()
        for distance in distances:
            filter2 = algo_df['Distance']  == distance
            algo_dist_df = algo_df.where(filter2).dropna()
            if len(algo_dist_df) != 0:
                dict_result_algo_dist[distance] = algo_dist_df.iloc[0].to_dict()
        results_algo_dist[algo] = dict_result_algo_dist
    
    best_k = best_overall['Number of clusters']
    sorted_df.to_csv(f'{main_path}/{path}')
    return best_k, best_overall, results_algos, results_algo_dist


def run_mountain(X, Y, dm, distance_definitions, main_path):

    sigmas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1]
    tols = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1]

    ras = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1]

    grid_points = [11]

    path = 'Results Mountain and Subtractive.csv'


    got_labels = Y is not None


    df_results = run_mountain_algorithms(X, Y, dm, distance_definitions, grid_points, tols, sigmas, ras, betas = None, rbs=None, main_path = main_path, include_external = got_labels)
    print(df_results)
    best_k , best_overall, results_algos, results_algo_dist = get_best_model(df_results, path, main_path = main_path, include_external = got_labels)
    return best_k


def run_k_means(X, Y, ns_clusters, distance_definitions, main_path):


    ms = [2]

    max_its = [100]
    tols = [1e-3]

    path = 'Results Kmeans and FuzzyCmeans.csv'

    got_labels = Y is not None

    df_results = run_kmeans_algorithms(X, Y,  distance_definitions,ns_clusters, ms, max_its, tols, main_path, include_external = got_labels)
    best_k , best_overall, results_algos, results_algo_dist = get_best_model(df_results, path, main_path, include_external = got_labels)
    return best_k


def run_all(X, Y, ns_clusters, distance_definitions, main_path):
    dm = DistanceMatrices(main_path)
    dm.compute_distance_matrices(X, distance_definitions)

    run_all_naive_algorithms(X, dm,  main_path= main_path)
    best_k = run_mountain(X, Y, dm, distance_definitions, main_path = main_path)
    run_k_means(X, Y, ns_clusters, distance_definitions, main_path = main_path)



if __name__ == '__main__':

    file_name = sys.argv.pop(1)
    main_path = sys.argv.pop(1)
    list_ks = sys.argv.pop(1)
    type= sys.argv.pop(1)

    list_ks = list_ks.split(',')
    ns_clusters = [int(k) for k in list_ks]
    X_norm, Y, inv_covmat = process_data(file_name)


    if type == 'low' or type == 'high':
        X_norm = find_best_low_high_dimension_data(type, X_norm)
        inv_covmat = np.linalg.inv(np.cov(X_norm, rowvar=False))
    elif type == 'umap':
        umap_args = ast.literal_eval(sys.argv.pop(1))
        #{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, "metric":'euclidean'}
        X_norm = UMAP_EMBEDDING(X_norm, umap_args).get_embedded_data()
        inv_covmat = np.linalg.inv(np.cov(X_norm, rowvar=False))

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
            'inv_covmat': inv_covmat
        },
        'Cosine':{
            'distance_name': 'Cosine',
        }
    }

    run_all(X_norm, Y, ns_clusters, distance_definitions, main_path = main_path)
