from utils import create_folders_if_not_exist,getDistinctColors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Graficas de clusters
class Cluster:
    def __init__(self, data, distance_matrix, algorithm, clustering_args, distance_name):

        create_folders_if_not_exist(['Results', 'Results/Clusters', f'Results/Clusters/{algorithm}/{distance_name}'])
        self.data = data
        self.distance_matrix = distance_matrix
        self.distance_name = distance_name
        self.algorithm = algorithm
        self.clustering_args = clustering_args
        params_strs = [f'{name} = {clustering_args[name]}'for name in clustering_args]
        self.name =  algorithm + ' with ' + distance_name + ' ' +  ','.join(params_strs)
        self.img_path = f'Results/Clusters/{algorithm}/{distance_name}/clusters {self.name}.png'
        self.memership_mat_path = f'Results/Clusters/{algorithm}/{distance_name}/memberships {self.name}.csv'
        self.cluster()
    
    def cluster(self):

        if self.algorithm == 'neighbours':
            self.groups = self.naive_k_neighbours_clustering(**self.clustering_args)
        if self.algorithm == 'boxes':
            self.groups = self.naive_box_clustering(**self.clustering_args)
        
        self.n_sets = len(self.groups)

    def naive_k_neighbours_clustering(self,k = 10):
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
        self.memebership_matrix = np.zeros((self.data.shape[0],len(self.groups)))
        for i,group in enumerate(self.groups):
            for j in group:
                self.memebership_matrix[j,i] = 1
        df = pd.DataFrame(self.memebership_matrix)
        df.to_csv(self.memership_mat_path)
        


    def graph_clusters(self, indexes = [0,1,2]):

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

