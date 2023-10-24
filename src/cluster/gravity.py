from utils import create_folders_if_not_exist,getDistinctColors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from distances import DistanceMatrices

class GravityClustering:
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

        N = self.data.shape[0]
        d_x_centers = DistanceMatrices().compute_distance_matrix(centers, self.data, **self.distance_args)**2
        index_min = np.argmin(d_x_centers, axis = 0)
        membership = np.zeros((len(centers), N))
        membership[index_min, np.arange(N)] = 1
        return membership
    

    def gravity_clustering(self, k, G0, p , epsilon,  T):
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


    def gravity_clustering_works2(self, k, G0, p , epsilon,  T):
        DM = DistanceMatrices()
        N, M = self.data.shape

        
        initial_indexes = np.random.choice(N, k, replace = False)
        centers = self.data[initial_indexes]
        velocities = np.zeros(shape=(k,M))
        t = 0
        while t <= T:

            d_centers = np.power(DM.compute_distance_matrix_fast(centers, self.data, **self.distance_args),2)
            minimum_distances_index = np.argmin(d_centers, axis = 0)
            
            memberships = np.zeros(shape=(k, N))
            memberships[minimum_distances_index, np.arange(N)] = 1

            Gt = G0* (1 - t/T)
            forces = np.zeros(shape=(k,M))
            for j in range(k):
                mj = 1
                mi = 1
                mass = mi * mj
                aux_indexes = np.where(memberships[j,:] == 1)[0]
                members = self.data[aux_indexes, :]
                Cj = len(members)

                diff = members - centers[j,:]
                Rij = d_centers[j,aux_indexes]
                denominator = 1/(d_centers[j,aux_indexes]**p + epsilon)
                S = 0
                if Cj > 0:
                    S = (Gt/Cj) *denominator.reshape(-1,1)*diff*np.random.rand(Cj, M)
                forces[j,:] = np.sum(S, axis = 0)

            new_velocities = velocities + forces
            centers = centers + new_velocities
            velocities = new_velocities
            t += 1
        self.centers = centers
        self.memberships = memberships

    def gravity_clustering_WORKS(self, k, G0, p , epsilon,  T):
        DM = DistanceMatrices()
        N, M = self.data.shape
        initial_indexes = np.random.choice(N, k, replace = False)
        initial_centers = self.data[initial_indexes]
        velocities = np.zeros((k,M))

        centers = initial_centers
        iteration = 0
        while iteration <= T:

            membership_matrix = np.zeros((k, N))
            distance_matrix = np.power(DM.compute_distance_matrix_fast(centers, self.data, **self.distance_args),2)
            minimum_distances_index = np.argmin(distance_matrix, axis = 0)
            cols = np.arange(N)
            membership_matrix[minimum_distances_index, cols] = 1

            g_t = G0*(1-iteration/T)
            forces = np.zeros((k,M))
            for j in range(k):
                aux_indexes = np.where(membership_matrix[j,:] == 1)[0]
                members = self.data[aux_indexes, :]
                n_members = len(members)
                diff = members - centers[j,:]
                denominator = 1/(distance_matrix[j,aux_indexes]**p + epsilon)
                sum_ = 0
                if n_members > 0:
                    sum_ = 1/n_members*denominator.reshape(-1,1)*diff*np.random.rand(n_members, M)
                forces[j,:] = np.sum(sum_, axis = 0)
            forces *= g_t
            new_velocities = velocities + forces

            centers = centers + new_velocities
            velocities = new_velocities
            iteration += 1
        self.centers = centers
        self.memberships = membership_matrix

    def gravity_clustering_new(self, k, G0, p , epsilon,  T):
        DM = DistanceMatrices()
        N, M = self.data.shape
        # Generate randomly positions of K initial agent (centroids) and set mass values of these objects to one
        velocities = np.zeros((k,M))
        centers = self.data[np.random.choice(N, k, replace = False)]
        i = 0
        while i <= T:
            # Assign each point to the closest center
            
            d_centers = np.power(DM.compute_distance_matrix_fast(centers, self.data, **self.distance_args),2)
            memberships = np.zeros((k, N))
            memberships[np.argmin(d_centers, axis = 0), np.arange(N)] = 1

            Gt = G0 * (1 - i/T)
            F = np.zeros(shape=(k,M))

            # Update centroids
            for j in range(k):
                indices_cluster = np.where(memberships[j,:] == 1)[0]
                members = self.data[indices_cluster, :]
                Cj = len(members)
                diff = members - centers[j,:]
                Rjp = d_centers[j, indices_cluster]**p
                denominator = 1/(Rjp + epsilon)
                partial_sum = 0
                if Cj > 0:
                    mass = np.random.rand(Cj, M)
                    partial_sum = 1/Cj*denominator.reshape(-1,1)*diff*mass
                F[j,:] = np.sum(partial_sum, axis = 0)
            
            F = Gt * F
            new_velocities = velocities + F
            new_centers = centers + new_velocities

            centers = new_centers
            velocities = new_velocities
            i += 1
        self.centers = centers
        self.memberships = memberships



    def save_results(self, indexes = [0,1,2]):
        self.graph_clusters(indexes)
        df = pd.DataFrame(self.memberships)
        df.to_csv(self.memership_mat_path)

    def get_groups(self):
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