from cluster.gravity import GravityClustering
from process import process_data
import sys
import numpy as np
from embedding import UMAP_EMBEDDING
from itertools import product
import ast
from cluster.validation import Validation
import pandas as pd


def rungravity(X,Y, ns_clusters, g0s,ps, epsilons,MAX_IS,  distance_definitions,main_path, include_external):

    indexes_results = []

    comb_gravity = list(product(ns_clusters, g0s, ps, epsilons, MAX_IS))
    keys_gravity = ('k', 'G0', 'p', 'epsilon', 'T')
    gravity_params = [ dict(zip(keys_gravity,tup)) for tup in comb_gravity]


    algo_name = 'gravity'

    for distance_id in distance_definitions:
        distance_args = distance_definitions[distance_id]
        distance_name = distance_args['distance_name']
        for clustering_args in gravity_params:
            gc = GravityClustering(X, algo_name, clustering_args, distance_args, distance_name, main_path = main_path)
            gc.cluster()
            gc.save_results()
            gc_val = Validation(gc.memberships, X, distance_args, centroids = None)
            internal_indexes = gc_val.get_all_internal_indexes()
            centers = gc_val.centroids
            # print(gc.centers.shape,centers.shape)
            if include_external:
                external_indexes = gc_val.get_all_external_indexes(Y)
                indexes_results.append((algo_name, gc.params_strs, distance_name, *internal_indexes.values() , *external_indexes.values(), len(centers)))
            else:
                indexes_results.append((algo_name, gc.params_strs, distance_name, *internal_indexes.values() , len(centers)))
    
    if include_external:  
        df_indices = pd.DataFrame(indexes_results, columns= ['Algorithm', 'Parameters', 'Distance', 'CH', 'BH', 'Hartigan', 'xu', 'DB', 'S', 'Rand', 'Fowlkes-Mallows', 'Jaccard', 'Number of clusters'])
    else:
        df_indices = pd.DataFrame(indexes_results, columns= ['Algorithm', 'Parameters', 'Distance', 'CH', 'BH', 'Hartigan', 'xu', 'DB', 'S', 'Number of clusters'])
    return df_indices