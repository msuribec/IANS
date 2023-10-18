
import numpy as np
from itertools import product
from cluster.naive import NaiveClustering
from process import CleanData
from distances import DistanceMatrices
import matplotlib.pyplot as plt
import pandas as pd
from utils import create_folders_if_not_exist



def get_params_for_naive_algortihm(distance_mat, n_intervals = 4, steps = 10, algorithms = ['neighbours', 'boxes']):
    """Returns a dictionary with the parameters for each algorithm
    Parameters:
        distance_mat (np.array):
            distance matrix of the data
        n_intervals (int):
            number of intervals to use in the boxes algorithm
        steps (int):
            number of steps to use in the neighbours algorithm
        algorithms (list):
            list of the algorithms to use
    Returns:
        algo_params (dict):
            dictionary with the parameters for each algorithms
    """

    min_d = np.min(distance_mat)
    max_d = np.max(distance_mat)

    
    N = distance_mat.shape[0]

    ks_values = [i*steps for i in range(1,int(N/steps))]
    thresholds = list(np.linspace(min_d,max_d,n_intervals+1))
    
    threshold_values = thresholds[1:-1]

    clustering_algorithms = {}

    for algo in algorithms:
        if algo == 'neighbours':
            clustering_algorithms[algo] = {
                'k' : ks_values
            }
        elif algo == 'boxes':
            
            clustering_algorithms[algo] = {
                'threshold' : threshold_values
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


def plot_neighbours_results(df, distances, main_path):

    df.plot(x="k", y=distances, kind="line")
    plt.title("Number of neighbours vs Number of clusters obtained")
    plt.ylabel("Number of clusters")
    plt.savefig(f'{main_path}/ParameterExploration/Neighbours.png')
    plt.close()

def plot_boxes_results(results_dict, distances, main_path):
    for distance in distances:
        plt.plot(*zip(*results_dict[distance]))
        plt.title(f"Radius of box vs Number of clusters obtained for the {distance} distance")
        plt.ylabel("Number of clusters")
        plt.savefig(f'{main_path}/ParameterExploration/Boxes {distance}.png')
        plt.close()

def plot_naive_results(neighbours_results, boxes_results, main_path = 'Iris'):
    results_dict = []
    distances = list(neighbours_results.keys())
    values_list = list(neighbours_results.values())
    tuples_list = values_list[0]
    n_ks = len(tuples_list)

    for i in range(n_ks):
        results_k = {'k': tuples_list[i][0]}

        for dist_name in neighbours_results:
            results_k[dist_name] = neighbours_results[dist_name][i][1]
        results_dict.append(results_k)
    neighbours_df = pd.DataFrame.from_records(results_dict)
    plot_neighbours_results(neighbours_df, distances, main_path)
    plot_boxes_results(boxes_results, distances, main_path)



# TODO: no hacer clean acaá
def run_all_naive_algorithms(data, dm, main_path = 'Iris'):

    """Runs all the clustering algorithms for the given data
    Parameters:
        data (np.array):
            array (N x M) of the data to cluster. N is the number of points in the dataset and M is the number of features
    """


    create_folders_if_not_exist([main_path, f'{main_path}/ParameterExploration'])
    dist_dictionary = dm.distance_matrices

    results_neighbours = {}
    results_boxes = {}

    results_overall = []

    for dist_name in dist_dictionary:

        dist_mat = dist_dictionary[dist_name]

        algo_dict = get_params_for_naive_algortihm(dist_mat)

        result_distance_neighbours = []
        result_distance_boxes = []

        for algo_name, algo_params_combinations in algo_dict.items():
            for algo_params in algo_params_combinations:
                c = NaiveClustering(data,dist_mat, algo_name,algo_params, dist_name, main_path)
                c.graph_clusters()
                c.get_membership()
                n_sets = c.n_sets
                results_overall.append((algo_name,dist_name,c.params_strs,c.n_sets))
                if algo_name == 'neighbours':
                    tup_results = (algo_params['k'] ,n_sets)
                    result_distance_neighbours.append(tup_results)
                if algo_name == 'boxes':
                    tup_results = (algo_params['threshold'], n_sets)
                    result_distance_boxes.append(tup_results)
                print(algo_name, algo_params, dist_name, n_sets)
        results_neighbours[dist_name] = result_distance_neighbours
        results_boxes[dist_name] = result_distance_boxes
    
    df_results = pd.DataFrame(results_overall, columns = ['Algorithm', 'Distance', 'Parameters', 'Number of clusters'])
    df_results.to_csv(f'{main_path}/ParameterExploration/Results Naive.csv', index = False)
    plot_naive_results(results_neighbours, results_boxes, main_path = main_path)
                    



