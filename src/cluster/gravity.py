from utils import create_folders_if_not_exist,getDistinctColors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from distances import DistanceMatrices

class GravityClustering:
    """Class that clusters data using the universal gravity rule algorithm
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
    def __init__(self, data, algorithm, clustering_args, distance_args, distance_name, main_path = 'Results'):

        self.main_path = main_path

        create_folders_if_not_exist([self.main_path, f'{self.main_path}/Clusters', f'{self.main_path}/Clusters/{algorithm}/{distance_name}'])
        self.data = data
        self.distance_args = distance_args

        self.distance_name = distance_name
        self.algorithm = algorithm
        self.clustering_args = clustering_args

        all_params = clustering_args.copy()

        # Name to put in the figure titles
        self.params_strs = ','.join([f'{name} = {all_params[name]}'for name in all_params])

        self.name =  algorithm + ' with ' + distance_name + ' ' +  self.params_strs

        # path where the 3d image of the clusters will be saved
        self.img_path = f'{self.main_path}/Clusters/{algorithm}/{distance_name}/clusters {self.name}.png'

        # Path where the membership matrix will be saved
        self.memership_mat_path = f'{self.main_path}/Clusters/{algorithm}/{distance_name}/memberships {self.name}.csv'
        self.cluster()
    
    def cluster(self):
        """Clusters the data using the specified algorithm and arguments"""

        if self.algorithm == 'gravity':
            self.gravity_clustering(**self.clustering_args)
        

    def get_memberships(self, centers):
        """Returns the membership matrix of the data given the centers
        Parameters:
            centers (numpy.ndarray):
                Centers of the clusters
        Returns:
            numpy.ndarray:
                Membership matrix of the data given the centers
        """

        N = self.data.shape[0]
        d_x_centers = DistanceMatrices().compute_distance_matrix(centers, self.data, **self.distance_args)**2
        index_min = np.argmin(d_x_centers, axis = 0)
        membership = np.zeros((len(centers), N))
        membership[index_min, np.arange(N)] = 1
        return membership
    

    def gravity_clustering(self, k, G0, p , epsilon,  T):
        """Clusters the data using the universal gravity rule algorithm
        Parameters:
            k (int):
                Number of clusters
            G0 (float):
                Gravitational constant
            p (float):
                Power
            epsilon (float):
                Epsilon
            T (int):
                Time (Number of iterations)
        """
        DM = DistanceMatrices()
        N, M = self.data.shape
        
        mi = 1
        mj = 1
        mass = mi * mj
        # Generate randomly positions of K initial agent (centroids) and set mass values of these objects to one
        centroid_indices = np.random.choice(N, k, replace = False)
        Z = self.data[centroid_indices] # Centers
        V = np.zeros(shape=(k,M)) # Velocities
        t = 0
        while t <= T:
            # Assign each point to the closest center

            d_centers = np.power(DM.compute_distance_matrix_fast(Z, self.data, **self.distance_args),2)

            memberships = np.zeros(shape=(k, N))
            memberships[np.argmin(d_centers, axis = 0), np.arange(N)] = 1
                
            Gt = G0 * (1 - t/T)
            F = np.zeros(shape=(k,M)) # Forces

            # Update cluster centroids
            for j in range(k):
                
                indices_cluster = np.where(memberships[j,:] == 1)[0]
                Xi = self.data[indices_cluster, :]
                Cj = len(Xi)
                Rij = d_centers[j,indices_cluster].reshape(-1,1)
                Fj = 0
                if Cj > 0:
                    ri = np.random.rand(Cj, M)
                    partial_sum =  ri* mass/(Rij**p + epsilon) * (Xi - Z[j,:])
                    Fj = (Gt/Cj) * np.sum(partial_sum, axis = 0)

                F[j,:] = Fj

            a = F/mj # Acceleration
            V += a
            Z += V
            t += 1

        self.centers = Z
        self.memberships = memberships


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

        groups = []
        for c in range(self.memberships.shape[0]):
            group_c = []
            for p in range(self.memberships.shape[1]):
                if self.memberships[c,p] == 1:
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


            self.groups = self.get_groups()


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