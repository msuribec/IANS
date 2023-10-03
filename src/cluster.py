from utils import create_folders_if_not_exist,getDistinctColors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from distances import DistanceMatrices


class Cluster:
    """Class that represents a cluster of points in a dataset
    Attributes:
        data (np.array):
            normalized array (N x M ) of the data to cluster. N is the number of points in the dataset and M is the number of features
        distance_matrix (np.array):
            distance matrix (N X N) where position i,j holds the distance between point i and point j
        distance_name (str):
            name  of the distance used to calculate the distance matrix
        algorithm (str):
            name of the clustering algorithm to use
        clustering_args (dict):
            arguments for the clustering algorithm
    """

    # TODO: change para que distance matrix sea opcional?
    def __init__(self, data, distance_matrix, algorithm, clustering_args, distance_name):

        create_folders_if_not_exist(['Results', 'Results/Clusters', f'Results/Clusters/{algorithm}/{distance_name}'])
        self.data = data
        self.distance_matrix = distance_matrix
        self.distance_name = distance_name
        self.algorithm = algorithm
        self.clustering_args = clustering_args

        # Name to put in the figure titles
        params_strs = [f'{name} = {clustering_args[name]}'for name in clustering_args]
        self.name =  algorithm + ' with ' + distance_name + ' ' +  ','.join(params_strs)

        # path where the 3d image of the clusters will be saved
        self.img_path = f'Results/Clusters/{algorithm}/{distance_name}/clusters {self.name}.png'

        # Path where the membership matrix will be saved
        self.memership_mat_path = f'Results/Clusters/{algorithm}/{distance_name}/memberships {self.name}.csv'
        self.cluster()
    
    def cluster(self):
        """Clusters the data using the specified algorithm and arguments"""

        if self.algorithm == 'neighbours':
            self.groups = self.naive_k_neighbours_clustering(**self.clustering_args)
        if self.algorithm == 'boxes':
            self.groups = self.naive_box_clustering(**self.clustering_args)
        
        self.n_sets = len(self.groups)

    def naive_k_neighbours_clustering(self,k = 10):
        """Clusters the data using the naive k neighbours algorithm
        Parameters:
            k (int):
                number of neighbours to use
        Returns:
            groups (list):
                list of lists. Each list contains the indexes of the points in the same cluster
        """
        N = self.distance_matrix.shape[0]

        classified_points = set()
        not_classified_points = set(range(N))
        groups = []

        i = 0
        it = 0
        while len(not_classified_points) > 0:
            
            index_sorted_distances = np.argsort(self.distance_matrix[i,:])
            close_points = index_sorted_distances[:k]
            groups.append(list(close_points))

            for j in close_points:
                not_classified_points.discard(j)
                classified_points.add(j)
            

            for j in index_sorted_distances[k:]:
                if j in not_classified_points:
                    i = j
                    break
            it += 1
        
        return groups


    def naive_box_clustering(self, threshold = 0.25):
        """Clusters the data using the naive box clustering algorithm
        Parameters:
            threshold (float):
                maximum distance between two points to be in the same cluster
        Returns:
            groups (list):
                list of lists. Each list contains the indexes of the points in the same cluster 
        """

        N = self.distance_matrix.shape[0]

        classified_points = set()
        not_classified_points = set(range(N))
        groups = []

        while len(not_classified_points) > 0:

            i = not_classified_points.pop()
            
            distances = self.distance_matrix[i,:]

            index_sorted_distances = np.argsort(distances)
            sorted_distances = distances[index_sorted_distances]
            close_points = set()
            for j in range(len(sorted_distances)):
                if sorted_distances[j] < threshold:
                    point = index_sorted_distances[j]
                    close_points.add(index_sorted_distances[point])
                    not_classified_points.discard(point)
                    classified_points.add(point)
                    
            groups.append(list(close_points))
        return groups
    
    def get_membership(self):
        """Saves the membership matrix of the clustering algorithm in a csv file
        The membership matrix is a (N X C) where N is the number of points and C the number of sets or clusters.
        position i,j of the matrix is 1 if point i belongs to set j and 0 otherwise
        """
        self.memebership_matrix = np.zeros((self.data.shape[0],len(self.groups)))
        for i,group in enumerate(self.groups):
            for j in group:
                self.memebership_matrix[j,i] = 1
        df = pd.DataFrame(self.memebership_matrix)
        df.to_csv(self.memership_mat_path)


    def get_memberships_from_centers(self,X, centers, distance_args):

        N = X.shape[0]
        d_x_centers = DistanceMatrices().compute_distance_matrix(centers, X, **distance_args)**2
        index_min = np.argmin(d_x_centers, axis = 0)
        membership = np.zeros((len(centers), N))
        membership[index_min, np.arange(N)] = 1
        return membership


    # TODO: TEST mountain clustering desde acá
    # TODO: integrar mountain clustering con el resto de los algoritmos
    def mountain_clustering(self,X, distance_mat_v, grid, sigma, distance_args , beta = None,  tolerance = 0.5):

        if beta is None:
            beta = 1.5*sigma

        DM = DistanceMatrices()
        c = 0
        n_points = grid.shape[0]
        mountain = np.zeros(n_points)

        while True:
        
            if c == 0:
                    d_vertex_x = distance_mat_v
                    mountain = np.sum(np.exp(-(d_vertex_x**2)/(2*sigma**2)),axis=0)
                    center = grid[np.argmax(mountain),:]
                    max_density = max(mountain)
                    first_max_density = max_density
                    centers = np.array([center])
            else:

                d_vertex_center  = DM.get_distance_vector(grid, center, **distance_args)
                aux = max_density*np.exp(-d_vertex_center**2/(2*beta**2))
                mountain = mountain - aux

                center = grid[np.argmax(mountain),:]
                max_density = max(mountain)

                center_in_list = np.any(np.equal(centers, center).all(axis=1))

                if center_in_list:
                    break
                
                centers = np.append(centers, center.reshape(1,-1), axis = 0)
                criterion = abs(max_density/first_max_density)
                if (criterion < tolerance):
                    break
            c += 1
        
        memberships = self.get_memberships_from_centers(X, centers, distance_args)
        return centers,memberships

    #TODO: integrar a otros algos
    # TODO: TEST subtractive clustering desde acá
    #TODO: tratar con rb = 2*ra

    def subtractive_clustering(self, X, distance_args, ra, tol, rb = None):

        if rb is None:
            rb = 1.5*ra

        DM = DistanceMatrices()
        
        count = 0 
        rep_centers = 0

        while True:
            if count == 0:
                matrix_distances  = DM.compute_distance_matrix_fast(X, X, **distance_args)
                mountain = np.sum(np.exp(-(matrix_distances**2)/((0.5*ra)**2))   ,axis=0)

                max_index = np.argmax(mountain)
                center = X[max_index,:]
                max_density = max(mountain)
                    
                first_max_density = max_density
                centers = np.array([center])
                tol = tol * abs(first_max_density)
            else:
                aux = max_density*np.exp(-(matrix_distances[:,max_index])**2 /((0.5*rb)**2))
                mountain = mountain - aux

                max_index = np.argmax(mountain)
                center = X[max_index,:]
                max_density = max(mountain)


                # if center in centers:
                #   break
                
                # centers = np.append(centers, center.reshape(1,-1), axis = 0)
                
                if center not in centers:
                    centers = np.append(centers, center.reshape(1,-1), axis = 0)
                else:
                    rep_centers += 1
                if rep_centers == 3:
                    # print("hum")
                    break
                if (abs(max_density) <= tol):
                    break
            count += 1
        memberships = self.get_memberships_from_centers(X, centers, distance_args)
        return centers, memberships


    def kmeans(self, x, k, distance_args, max_iter=100, tol=0.001):

        N, _ = x.shape
        dm = DistanceMatrices()
        costs = []
        initial_centers = np.random.choice(N, k, replace=False)
        centroids = x[initial_centers, :]
        i = 0
        while i < max_iter:

            membership_matrix = np.zeros((N, k), dtype=np.int32)
            distance_to_clusters = dm.compute_distance_matrix_fast(x, centroids, **distance_args)
            indices_min_dist_to_clusters = np.argmin(distance_to_clusters, axis=1)
            
            for j in range(N):
                membership_matrix[j, indices_min_dist_to_clusters[j]] = 1

            cost = np.sum(np.multiply(membership_matrix, distance_to_clusters))
            costs.append(cost)

            if len(costs) > 1 and abs(cost - costs[i-1]) < tol:
                break

            for c in range(k):
                indices = membership_matrix[:, c] == 1
                centroids[c, :] = np.mean(x[indices,:], axis=0)

            i= i+1

        return centroids, membership_matrix
    

    def fuzzykmeans(self, x, distance_args, N_CLUSTERS, m=2, MAX_I=100, tol=1e-3):
        dm = DistanceMatrices()
        
        N, M = x.shape

        fuzzy_membership = np.random.rand(N,N_CLUSTERS)
        fuzzy_membership = fuzzy_membership / np.sum(fuzzy_membership, axis=1, keepdims=True)

        i = 0
        centers = np.zeros(shape=(N_CLUSTERS,M))
        costs = []
        while i < MAX_I:

            denominator =  np.sum(fuzzy_membership ** m, axis = 0)

            for c in range(N_CLUSTERS):
                numerator = np.matmul(fuzzy_membership[:, c] ** m, x)
                centers[c, :] = numerator / denominator[c]

            distance_to_clusters = dm.compute_distance_matrix_fast(x, centers, **distance_args)
            cost = np.sum( np.multiply(fuzzy_membership** m , distance_to_clusters**2))
            costs.append(cost)

            if len(costs) > 1 and abs(cost-costs[i-1]) < tol:
                break

            for p in range(N):
                for c in range(N_CLUSTERS):
                    div_distances = np.divide(distance_to_clusters[p,c], distance_to_clusters[p,:])
                    fuzzy_membership[p,c] = 1 / np.sum(div_distances ** (2/(m-1)))

            i = i + 1

        return centers, fuzzy_membership


    def graph_clusters(self, indexes = [0,1,2]):
        """Graphs the clusters in a 3d plot and saves it to the path specified in the constructor.
        The graph will have the name specified in the constructor and only the features specified in the parameter will be graphed.
        Parameters:
            indexes (list):
                list of indexes of the features to graph. Must be of length 3
            
        """

        M = self.data.shape[1]

        assert M >= 3, "Data must have 3 dimensions at least"
        assert len(indexes) == 3, "Only pass the indexes of 3 features to graph"


        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(111, projection='3d')

        colors = list(getDistinctColors(len(self.groups)))

        for i,group in enumerate(self.groups):

            data_group = self.data[group, :]
                
            x = data_group[:, indexes[0]]
            y = data_group[:, indexes[1]]
            z = data_group[:, indexes[2]]

            alts = np.array([1.2] * len(group))

            ax.scatter(x,y,z, s=alts * 5, color = [colors[i]], cmap = None, depthshade=False, label=("Cluster " + str(i + 1)))

        plt.title('clusters ' + self.name)
        if len(self.groups) > 5:
            plt.legend(numpoints=1 , bbox_to_anchor=(1.1, 1.05))
        else:
            plt.legend(numpoints=1 , loc='best')
        plt.savefig(self.img_path)
        plt.close()

