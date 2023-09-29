from utils import ReadData
import numpy as np
from itertools import product
from cluster import Cluster
from process import CleanData
from distances import DistanceMatrices

def get_params_for_algortihm(distance_mat, n_intervals = 4, steps = 10, algorithms = ['neighbours', 'boxes']):

    min_d = np.min(distance_mat)
    max_d = np.max(distance_mat)

    N = distance_mat.shape[0]

    clustering_algorithms = {}

    for algo in algorithms:
        if algo == 'neighbours':
            clustering_algorithms[algo] = {
                'k' : [i*steps for i in range(1,int(N/steps))]
            }
        elif algo == 'boxes':
            thresholds = list(np.linspace(min_d,max_d,n_intervals+1))
            clustering_algorithms[algo] = {
                'threshold' : thresholds[1:-1]
            }


    algo_params = {}
    for algo_name,params_dict in clustering_algorithms.items():
        param_names = []
        param_values = []
        for param_name,param_value in params_dict.items():
            param_names.append(param_name)
            param_values.append(param_value)

        param_combinations = list(product(*param_values))

        list_dics = [ dict(zip(param_names,params)) for params in param_combinations]

        algo_params[algo_name] = list_dics

    return algo_params

def run_all_clustering_algorithms(raw_data):


    cd = CleanData(raw_data)
    data = cd.norm_data
    dm = DistanceMatrices()


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
            'inv_covmat': cd.inv_covmat
        },
        'Cosine':{
            'distance_name': 'Cosine',
        }
    }
    
    dm.compute_distance_matrices(data, distance_definitions)
    dist_dictionary = dm.distance_matrices

    for dist_name in dist_dictionary:

        dist_mat = dist_dictionary[dist_name]

        algo_dict = get_params_for_algortihm(dist_mat)
        for algo_name, algo_params_combinations in algo_dict.items():
            for algo_params in algo_params_combinations:
                c = Cluster(data,dist_mat, algo_name,algo_params, dist_name)
                c.graph_clusters()
                c.get_membership()
                n_sets = c.n_sets
                print(algo_name,dist_name,algo_params,n_sets)


if __name__ == '__main__':
    reader = ReadData()
    data_df = reader.read_file('Data/iris.csv')
    raw_data = data_df.drop(['variety'], axis=1).to_numpy()
    run_all_clustering_algorithms(raw_data)

