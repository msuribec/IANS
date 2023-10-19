from cluster.kmeans import KmeansCluster
from distances import DistanceMatrices
from itertools import product
from cluster.mountain import MountainClustering
import numpy as np
from cluster.validation import Validation
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



def run_kmeans_algorithms(X, Y,  distance_definitions,ns_clusters, ms, max_its, tols, main_path = 'Iris', include_external = True):
    indexes_results = []

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
                if include_external:
                    external_indexes = kc_val.get_all_external_indexes(Y)

                print(algo_name, clustering_args, dist_name, len(kc.centers))
                if include_external:
                    indexes_results.append((algo_name, kc.params_strs, dist_name, *internal_indexes.values() , *external_indexes.values() , len(kc.centers)) )
                else:
                    indexes_results.append((algo_name, kc.params_strs, dist_name, *internal_indexes.values() , len(kc.centers)) )

    if include_external:
        indices_cols = ['Algorithm', 'Parameters', 'Distance', 'CH', 'BH', 'Hartigan', 'xu', 'DB', 'S', 'Rand', 'Fowlkes-Mallows', 'Jaccard', 'Number of clusters']
    else:
        indices_cols = ['Algorithm', 'Parameters', 'Distance', 'CH', 'BH', 'Hartigan', 'xu', 'DB', 'S', 'Number of clusters']
    df_indices = pd.DataFrame(indexes_results, columns= indices_cols)
    return df_indices