from cluster.kmeans import KmeansCluster
from distances import DistanceMatrices
from itertools import product
from cluster.mountain import MountainClustering
import numpy as np
from cluster.validation import Validation
from sklearn import datasets
import pandas as pd

def get_params_kmeans_algorithms(ns_clusters, ms, max_its, tols):

  kmeans_params = {}

  comb_kmeans = list(product(ns_clusters, max_its, tols))
  comb_fcmeans = list(product(ns_clusters, ms, max_its, tols))

  keys_kmeans = ('k', 'MAX_I', 'tol')

  keys_fcmeans = ('k', 'm', 'MAX_I', 'tol')

  kmeans_params['kmeans'] = [ dict(zip(keys_kmeans,tup)) for tup in comb_kmeans]
  kmeans_params['fuzzycmeans'] = [ dict(zip(keys_fcmeans,tup)) for tup in comb_fcmeans]
  return kmeans_params



def run_kmeans_algorithms(X, Y,  distance_definitions,ns_clusters, ms, max_its, tols, main_path = 'Iris'):


    indexes_results = []
    internal_indexes_results = []
    external_indexes_results = []
    number_clusters = []
    #TODO: guardar en un dataframe los resultados de cada algoritmo
    # coger self.params_str
    mountain_algos_params = get_params_kmeans_algorithms(ns_clusters, ms, max_its, tols)

    dm = DistanceMatrices(main_path)

    for dist_id in distance_definitions:

        dist_args = distance_definitions[dist_id]
        dist_name = dist_args['distance_name']

        for algo_name in mountain_algos_params:
            for clustering_args in mountain_algos_params[algo_name]:

                kc = KmeansCluster(X, algo_name, clustering_args, dist_args, dist_name, main_path = main_path)
                kc.cluster()
                kc.save_results()

                kc_val = Validation(kc.memberships, X, dist_args, centroids = kc.centers)

                internal_indexes = kc_val.get_all_internal_indexes()
                external_indexes = kc_val.get_all_external_indexes(Y)

                print(algo_name, clustering_args, dist_name, len(kc.centers))
                indexes_results.append((algo_name, kc.params_strs, dist_name, *internal_indexes.values() , *external_indexes.values()))
                internal_indexes_results.append((algo_name, kc.params_strs, dist_name, *internal_indexes.values()))
                external_indexes_results.append((algo_name, kc.params_strs, dist_name, *external_indexes.values()))
                number_clusters.append((algo_name, kc.params_strs, dist_name, len(kc.centers)))

    df_indices = pd.DataFrame(indexes_results, columns= ['Algorithm', 'Parameters', 'Distance', 'CH', 'BH', 'Hartigan', 'xu', 'DB', 'S', 'Rand', 'Fowlkes-Mallows', 'Jaccard'])
    df_internal_indices = pd.DataFrame(internal_indexes_results, columns= ['Algorithm', 'Parameters', 'Distance', 'CH', 'BH', 'Hartigan', 'xu', 'DB', 'S'])
    df_external_indices = pd.DataFrame(external_indexes_results, columns= ['Algorithm', 'Parameters', 'Distance', 'Rand', 'Fowlkes-Mallows', 'Jaccard'])
    df_nclusters = pd.DataFrame(number_clusters, columns= ['Algorithm', 'Parameters', 'Distance', 'Number of clusters'])


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

    ns_clusters = [2,3,4]
    ms = [2]

    max_its = [100]
    tols = [1e-3]

    run_kmeans_algorithms(X, Y,  distance_definitions,ns_clusters, ms, max_its, tols, main_path = 'Iris')



def normalize_data(X):
  x_min = np.min(X, axis = 0)
  x_max = np.max(X, axis = 0)
  return (X-x_min)/(x_max-x_min)


if __name__ == '__main__':
  iris = datasets.load_iris()
  X = iris.data
  Y = iris.target

  X_norm = normalize_data(iris.data)

#   test_euclidean(X_norm)

  test_run(X_norm,Y)
 