
from cluster.mountain import MountainClustering
from cluster.kmeans import KmeansCluster
from process import CleanData
from distances import DistanceMatrices
from autoencoder import find_best_low_high_dimension_data
from run_density_based import get_vertices_grid
from embedding import UMAP_EMBEDDING
from utils import ReadData, create_folders_if_not_exist, getDistinctColors
from cluster.validation import Validation
from process import process_data
import sys
import ast
import numpy as np
import json
import matplotlib.pyplot as plt
import numpy as n

def run_best_mountain(X, Y, algo_name, clustering_args, dist_args, main_path, include_external = True):

    dist_name = dist_args['distance_name']

    dm = DistanceMatrices(main_path)
    dist_mat = dm.compute_distance_matrix_fast(X, X, **dist_args)

    if algo_name == 'subtractive':
        mc = MountainClustering(X, algo_name, clustering_args, dist_args, dist_name, distance_mat = dist_mat, main_path = main_path)
        mc.cluster()
    if algo_name == 'mountain':
        gp = clustering_args.pop('gridp')
        grid = get_vertices_grid(gp,X.shape[1])
        distance_mat_v = dm.compute_distance_matrix_fast(X, grid, **dist_args)
        mc = MountainClustering(X, algo_name, clustering_args, dist_args, dist_name, distance_mat_v = distance_mat_v, grid=grid, grid_points = gp, main_path = main_path)
        mc.cluster()

    predictions = np.argmax(mc.memberships, axis=0)

    kc_val = Validation(mc.memberships, X, dist_args, centroids = mc.centers)
    internal_indexes = kc_val.get_all_internal_indexes()
    if include_external:
        external_indexes = kc_val.get_all_external_indexes(Y)
        all_indexes = [*internal_indexes.values() , *external_indexes.values()]
    else:
        all_indexes = internal_indexes.values()
    
    return predictions, all_indexes

def run_kmeans(X, Y, algo_name, clustering_args, dist_args, main_path, include_external = True):
    
    dist_name = dist_args['distance_name']

    kc = KmeansCluster(X, algo_name, clustering_args, dist_args, dist_name, main_path = main_path)
    kc.cluster()

    kc_val = Validation(kc.memberships, X, dist_args, centroids = kc.centers)

    predictions = np.argmax(kc.memberships, axis=0)

    kc_val = Validation(kc.memberships, X, dist_args, centroids = kc.centers)
    internal_indexes = kc_val.get_all_internal_indexes()
    if include_external:
        external_indexes = kc_val.get_all_external_indexes(Y)
        all_indexes = [*internal_indexes.values() , *external_indexes.values()]
    else:
        all_indexes = internal_indexes.values()
    return predictions, all_indexes

def read_json(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
    return data


def plot_scatter_2d_clusters(data, labels, path):

    classes = np.unique(labels)

    colors = list(getDistinctColors(len(classes)))

    # Plot
    fig, ax = plt.subplots()
    for i,class_ in enumerate(classes):
      data_class = data[predictions==class_,:]
      labs = predictions[predictions==class_]
      clrs = [colors[i]] * len(data_class)
      ax.scatter(data_class[:, 0], data_class[:, 1], c=clrs, label=("Cluster " + str(i + 1)))
    if len(classes) > 5:
      plt.legend(numpoints=1 , bbox_to_anchor=(1.1, 1.05))
    else:
      plt.legend(numpoints=1 , loc='best')
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':

    file_name = sys.argv.pop(1)
    main_path = sys.argv.pop(1)
    type= sys.argv.pop(1)
    json_file_name = sys.argv.pop(1)
    umap_args = ast.literal_eval(sys.argv.pop(1))

    create_folders_if_not_exist([main_path, main_path + '/Best'])

    X_norm, Y, inv_covmat = process_data(file_name)

    got_labels = Y is not None

    if type == 'low' or type == 'high':
        X_norm = find_best_low_high_dimension_data(type, X_norm, Y)
        inv_covmat = np.linalg.inv(np.cov(X_norm, rowvar=False))
    elif type == 'umap':
        #{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, "metric":'euclidean'}
        X_norm = UMAP_EMBEDDING(X_norm, umap_args).get_embedded_data()
        inv_covmat = np.linalg.inv(np.cov(X_norm, rowvar=False))


    json_best_algos = read_json(json_file_name)
    
    for algo_name in json_best_algos:
        type_definitions = json_best_algos[algo_name]
        for type in type_definitions:
            definition = type_definitions[type]
            name = definition['name']
            dist_args = definition['dist_args']
            if dist_args['distance_name'] == 'Mahalanobis':
                dist_args['inv_covmat'] = inv_covmat
            clustering_args = definition['clustering_args']

            if algo_name in ['mountain', 'subtractive']:

                predictions, all_indexes = run_best_mountain(X_norm, Y, algo_name, clustering_args, dist_args ,main_path, include_external = got_labels)
            else:
                predictions, all_indexes = run_kmeans(X_norm, Y, algo_name, clustering_args, dist_args, main_path, include_external = got_labels)
            path = f'{main_path}/Best/{name}.png'

            if X_norm.shape[1] == 2:
                
                plot_scatter_2d_clusters(X_norm, predictions, path)
            else:
                UMAP_EMBEDDING(X_norm, umap_args).plot_umap(predictions, path)