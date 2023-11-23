from utils import create_folders_if_not_exist,getDistinctColors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from distances import DistanceMatrices

class MountainClustering:
    """Class that clusters data using the mountain clustering algorithm
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
        distance_mat (numpy.ndarray):
            Distance matrix of the data
        distance_mat_v (numpy.ndarray):
            Distance matrix of the vertices
        grid (numpy.ndarray):
            Grid of the vertices
        grid_points (int):
            Number of points in the grid
        main_path (str):
            Path where the results will be saved
    """
    def __init__(self, data, algorithm, clustering_args, distance_args, distance_name, distance_mat = None,  distance_mat_v = None, grid = None,grid_points = None, main_path = 'Results'):

        self.main_path = main_path

        create_folders_if_not_exist([self.main_path, f'{self.main_path}/Clusters', f'{self.main_path}/Clusters/{algorithm}/{distance_name}'])
        self.data = data
        self.distance_mat = distance_mat
        self.distance_args = distance_args
        self.distance_mat_v = distance_mat_v
        self.grid = grid
        self.distance_name = distance_name
        self.algorithm = algorithm
        self.clustering_args = clustering_args

        all_params = clustering_args.copy()

        if algorithm == 'mountain':
            all_params['grid p'] = grid_points
            assert grid is not None, "Grid must be provided for mountain clustering"

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

        if self.algorithm == 'mountain':
            self.mountain_clustering(**self.clustering_args)
        if self.algorithm == 'subtractive':
            self.subtractive_clustering(**self.clustering_args)
        

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

    def mountain_clustering(self, sigma , beta = None,  tol = 0.5):

        if beta is None:
            beta = 1.5*sigma

        DM = DistanceMatrices()
        c = 0
        n_points = self.grid.shape[0]
        mountain = np.zeros(n_points)

        while True:
            if c == 0:
                    d_vertex_x = self.distance_mat_v
                    mountain = np.sum(np.exp(-(d_vertex_x**2)/(2*sigma**2)),axis=0)
                    center = self.grid[np.argmax(mountain),:]
                    max_density = max(mountain)
                    tol = tol * abs(max_density)
                    centers = np.array([center])
            else:

                d_vertex_center  = DM.get_distance_vector(self.grid, center, **self.distance_args)
                aux = max_density*np.exp(-d_vertex_center**2/(2*beta**2))
                mountain = mountain - aux

                center = self.grid[np.argmax(mountain),:]
                max_density = max(mountain)

                center_in_list = np.any(np.equal(centers, center).all(axis=1))

                if center_in_list:
                    break
                
                centers = np.append(centers, center.reshape(1,-1), axis = 0)
                if (abs(max_density) <= tol):
                    break
            c += 1
        
        memberships = self.get_memberships(centers)
        self.centers = centers
        self.memberships = memberships


    def subtractive_clustering(self, ra, tol, rb = None):
        """Clusters the data using the subtractive clustering algorithm
        Parameters:
            ra (float):
                Radius of the first mountain
            tol (float):
                Tolerance
            rb (float):
                Radius of the second mountain
        """

        if rb is None:
            rb = 1.5*ra

        DM = DistanceMatrices()
        
        count = 0 
        rep_centers = 0

        while True:
            if count == 0:

                if self.distance_mat is None:
                    self.distance_mat  = DM.compute_distance_matrix_fast(self.data, self.data, **self.distance_args)
                
                mountain = np.sum(np.exp(-(self.distance_mat**2)/((0.5*ra)**2))   ,axis=0)

                max_index = np.argmax(mountain)
                center = self.data[max_index,:]
                max_density = max(mountain)
                    
                first_max_density = max_density
                centers = np.array([center])
                tol = tol * abs(first_max_density)
            else:
                aux = max_density*np.exp(-(self.distance_mat[:,max_index])**2 /((0.5*rb)**2))
                mountain = mountain - aux

                max_index = np.argmax(mountain)
                center = self.data[max_index,:]
                max_density = max(mountain)

                
                if center not in centers:
                    centers = np.append(centers, center.reshape(1,-1), axis = 0)
                else:
                    rep_centers += 1
                if rep_centers == 3:

                    break
                if (abs(max_density) <= tol):
                    break
            count += 1
        memberships = self.get_memberships(centers)
        self.centers = centers
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