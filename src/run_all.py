from run_naive import run_all_naive_algorithms
from run_density_based import run_mountain_algorithms
from runkmeans import run_kmeans_algorithms
from process import CleanData
from distances import DistanceMatrices
from utils import ReadData
import sys
import numpy as np



def get_best_model(df, main_path = 'Iris'):
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
            dict_result_algo_dist[distance] = algo_dist_df.iloc[0].to_dict()
        results_algo_dist[algo] = dict_result_algo_dist
    
    best_k = best_overall['Number of clusters']
    sorted_df.to_csv(f'{main_path}/Results Mountain and Subtractive.csv')
    return best_k, best_overall, results_algos, results_algo_dist



def run_mountain(X, Y, distance_definitions, main_path = 'Iris'):

    # sigmas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1]
    # tols = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1]

    # ras = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1]

    sigmas = [0.1,0.2]
    tols = [0.5,0.6]

    ras = [0.1,0.2]

    df_results = run_mountain_algorithms(X, Y, distance_definitions, [6, 11], tols, sigmas, ras, betas = None, rbs=None, main_path = main_path)
    best_k , best_overall, results_algos, results_algo_dist = get_best_model(df_results, main_path = main_path)
    return best_k

def run_k_means(X, Y, distance_definitions, main_path = 'Iris'):

    ns_clusters = [2,3,4]
    ms = [2]

    max_its = [100]
    tols = [1e-3]

    run_kmeans_algorithms(X, Y,  distance_definitions,ns_clusters, ms, max_its, tols, main_path = main_path)


def process_data(filename):

    reader = ReadData()
    
    data_df = reader.read_file(file_name)
    X = data_df.drop(['target'], axis=1).to_numpy()
    N, M = X.shape
    target_values = data_df['target'].values
    unique_classes = np.unique(target_values)
    n_classes = len(unique_classes)
    Y = np.zeros(N, dtype=np.int32)
    for i,target in enumerate(target_values):
        for j in range(n_classes):
            if target == unique_classes[j]:
                Y[i] = j
    
    cd = CleanData(X)
    X_norm = cd.norm_data
    inv_covmat = cd.inv_covmat

    return X_norm, Y, inv_covmat

if __name__ == '__main__':

    file_name = sys.argv.pop(1)
    main_path = sys.argv.pop(1)


    X_norm, Y, inv_covmat = process_data(file_name)

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
    dm = DistanceMatrices(main_path)
    dm.compute_distance_matrices(X_norm, distance_definitions)

    # run_all_naive_algorithms(X_norm,dm,  main_path= main_path)
    # best_k = run_mountain(X_norm, Y, dm, main_path = main_path)

    run_k_means(X_norm, Y, distance_definitions, main_path = main_path)
