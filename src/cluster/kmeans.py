from utils import create_folders_if_not_exist,getDistinctColors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from distances import DistanceMatrices


class KmeansCluster:
    """Class that clusters data using the kmeans and fuzzy cmeans algorithms
    Parameters:
        data (numpy.ndarray):
            Data to cluster
        algorithm (str):
            Name of the clustering algorithm
        clustering_args (dict):
            Dictionary with the arguments of the clustering algorithm
        distance_args (dict):
            Dictionary with the arguments of the distance function
        distance_name (str):
            Name of the distance function
        main_path (str):
            Path where the results will be saved
    """
    def __init__(self, data, algorithm, clustering_args, distance_args,distance_name, main_path = 'Results'):

        self.main_path = main_path

        create_folders_if_not_exist([self.main_path, f'{self.main_path}/Clusters', f'{self.main_path}/Clusters/{algorithm}/{distance_name}'])
        self.data = data
        self.distance_args = distance_args
        self.distance_name = distance_name
        self.algorithm = algorithm
        self.clustering_args = clustering_args

        # Name to put in the figure titles
        self.params_strs = ','.join([f'{name} = {clustering_args[name]}'for name in clustering_args])
        self.name =  algorithm + ' with ' + distance_name + ' ' + self.params_strs

        # path where the 3d image of the clusters will be saved
        self.img_path = f'{self.main_path}/Clusters/{algorithm}/{distance_name}/clusters {self.name}.png'

        # Path where the membership matrix will be saved
        self.memership_mat_path = f'{self.main_path}/Clusters/{algorithm}/{distance_name}/memberships {self.name}.csv'
        self.cluster()
    
    def cluster(self):

        """Clusters the data using the specified algorithm and arguments"""

        if self.algorithm == 'kmeans':
            self.kmeans(**self.clustering_args)
        if self.algorithm == 'fuzzycmeans':
            self.fuzzyCmeans(**self.clustering_args)
        
        

    def kmeans(self, k=3, MAX_I=100, tol=1e-3):
        """Clusters the data using the kmeans algorithm
        Parameters:
            k (int):
                Number of clusters
            MAX_I (int):
                Maximum number of iterations
            tol (float):
                Tolerance
        Returns:
            numpy.ndarray:
                Centers of the clusters
            numpy.ndarray:
                Membership matrix of the data given the centers
        """

        N, _ = self.data.shape
        dm = DistanceMatrices()
        costs = []
        initial_centers = np.random.choice(N, k, replace=False)
        centroids = self.data[initial_centers, :]
        i = 0
        while i < MAX_I:
            membership_matrix = np.zeros((k, N), dtype=np.int32)

            distance_to_clusters = dm.compute_distance_matrix_fast(centroids,self.data, **self.distance_args)
            indices_min_dist_to_clusters = np.argmin(distance_to_clusters, axis=0)
            
            for j in range(N):
                membership_matrix[indices_min_dist_to_clusters[j], j] = 1

            cost = np.sum(np.multiply(membership_matrix, distance_to_clusters))
            costs.append(cost)

            if len(costs) > 1 and abs(cost - costs[i-1]) < tol:
                break

            for c in range(k):
                indices = membership_matrix[c, :] == 1
                centroids[c, :] = np.mean(self.data[indices,:], axis=0)

            i= i+1

        self.centers = centroids
        self.memberships = membership_matrix
        return centroids, membership_matrix


    def fuzzyCmeans(self, k=3, m=2, MAX_I=100, tol=1e-3, verbose=True):
        """Clusters the data using the fuzzy cmeans algorithm
        Parameters:
            k (int):
                Number of clusters
            m (int):
                Fuzziness parameter
            MAX_I (int):
                Maximum number of iterations
            tol (float):
                Tolerance
            verbose (bool):
                If true prints the iteration and cost
        Returns:
            numpy.ndarray:
                Centers of the clusters
            numpy.ndarray:
                Membership matrix of the data given the centers
        """
        dm = DistanceMatrices()
        
        N, M = self.data.shape
        fuzzy_membership = np.random.rand(N, k)
        fuzzy_membership = fuzzy_membership.T

        fuzzy_membership = fuzzy_membership / np.sum(fuzzy_membership, axis=0, keepdims=True)

        i = 0
        centers = np.zeros(shape=(k, M))
        costs = []
        while i < MAX_I:

            denominator =  np.sum(fuzzy_membership ** m, axis = 1)

            for c in range(k):
                numerator = np.matmul(fuzzy_membership[c,:] ** m, self.data)
                centers[c, :] = numerator / denominator[c]

            distance_to_clusters = dm.compute_distance_matrix_fast(centers,self.data, **self.distance_args)
            cost = np.sum( np.multiply(fuzzy_membership** m , distance_to_clusters**2))
            costs.append(cost)

            if len(costs) > 1 and abs(cost-costs[i-1]) < tol:
                break

            for p in range(N):
                for c in range(k):
                    div_distances = np.divide(distance_to_clusters[c,p], distance_to_clusters[:,p])
                    fuzzy_membership[c,p] = 1 / np.sum(div_distances ** (2/(m-1)))

            i = i + 1

            # if verbose:
            #     print(f'Iteration: {i}, Cost: {cost}')

        self.centers = centers
        self.memberships = fuzzy_membership
        return centers, fuzzy_membership
    
    def save_results(self, indexes = [0,1,2]):
        """Saves the membership matrix and graphs the clusters in a 3d plot.
        The graph will have the name specified in the constructor and only the features specified in the parameter will be graphed.
        Parameters:
            indexes (list):
                list of indexes of the features to graph. Must be of length 3
            
        """
        self.graph_clusters(indexes)
        df = pd.DataFrame(self.memberships)
        df.to_csv(self.memership_mat_path)

    
    def get_groups(self):
        """Returns a list of lists with the indexes of the data points in each cluster"""

        members = np.max(self.memberships, axis=0)
        groups = []
        for c in range(self.memberships.shape[0]):
            group_c = []
            for p in range(self.memberships.shape[1]):
                if self.memberships[c,p] == members[p]:
                    group_c.append(p)
            groups.append(group_c)
        
        return groups


    def graph_clusters(self, indexes = [0,1,2]):
        """Graphs the clusters in a 3d plot and saves it to the path specified in the constructor.
        The graph will have the name specified in the constructor and only the features specified in the parameter will be graphed.
        Parameters:
            indexes (list):
                list of indexes of the features to graph. Must be of length 3
            
        """

        M = self.data.shape[1]

        assert len(indexes) == 3, "Only pass the indexes of 3 features to graph"

        if M == 2:
            self.graph_cluster_2d(indexes)
        else:


            fig = plt.figure(figsize = (10, 10))
            ax = fig.add_subplot(111, projection='3d')


            groups = self.get_groups()


            colors = list(getDistinctColors(len(groups)))

            for i,group in enumerate(groups):

                data_group = self.data[group, :]
                    
                x = data_group[:, indexes[0]]
                y = data_group[:, indexes[1]]
                z = data_group[:, indexes[2]]

                alts = np.array([1.2] * len(group))

                ax.scatter(x,y,z, s=alts * 5, color = [colors[i]], cmap = None, depthshade=False, label=("Cluster " + str(i + 1)))

            plt.title('clusters ' + self.name)
            if len(groups) > 5:
                plt.legend(numpoints=1 , bbox_to_anchor=(1.1, 1.05))
            else:
                plt.legend(numpoints=1 , loc='best')
            plt.savefig(self.img_path)
            plt.close()


    def graph_cluster_2d(self, indexes = [0,1]):
        """Graphs the clusters in a 2d plot and saves it to the path specified in the constructor.
        The graph will have the name specified in the constructor and only the features specified in the parameter will be graphed.
        Parameters:
            indexes (list):
                list of indexes of the features to graph. Must be of length 2
            
        """
        
    
        M = self.data.shape[1]

        assert M == 2, "Data must have 2 dimensions at least"

        self.groups = self.get_groups()

        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(111)

        colors = list(getDistinctColors(len(self.groups)))

        for i,group in enumerate(self.groups):

            data_group = self.data[group, :]
                
            x = data_group[:, indexes[0]]
            y = data_group[:, indexes[1]]

            alts = np.array([1.2] * len(group))

            ax.scatter(x,y, s=alts * 5, color = [colors[i]], cmap = None, label=("Cluster " + str(i + 1)))

        plt.title('clusters ' + self.name)
        if len(self.groups) > 5:
            plt.legend(numpoints=1 , bbox_to_anchor=(1.1, 1.05))
        else:
            plt.legend(numpoints=1 , loc='best')
        plt.savefig(self.img_path)
        plt.close()


