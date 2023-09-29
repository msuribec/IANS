from utils import create_folders_if_not_exist
import numpy as np
import matplotlib.pyplot as plt
 

class DistanceMatrices:

    def __init__(self):
        pass

    def compute_distance_matrices(self, data, distance_definitions):
        create_folders_if_not_exist(['Results', 'Results/Distance Matrices','Results/Distance to vertices', 'Results/Classifications'])
        self.distance_matrices = {}
        for distance_id in distance_definitions:
            params = distance_definitions[distance_id]
            dist_matrix = self.compute_distance_matrix(data,**params)
            self.distance_matrices[distance_id] = dist_matrix
            self.graph_distance_matrix(dist_matrix, distance_id, False)

    def get_distance(self,v1,v2,distance, p=None, inv_covmat = None):
        if distance == 'Euclidean':
            dist_v1v2 = self.minkowski(v1,v2, 2)
        elif distance == 'Manhattan':
            dist_v1v2 =  self.minkowski(v1,v2, 1)
        elif distance == 'Minkowski':
            assert p is not None, "p must be specified when using p_distance"
            dist_v1v2 =  self.minkowski(v1,v2, p)
        elif distance == 'Mahalanobis':
            dist_v1v2 = self.mahalanobis(v1,v2,inv_covmat)   
        elif distance == 'Cosine':
            dist_v1v2 = self.cosine_distance(v1,v2)
        return dist_v1v2
        
    def compute_distance_matrix(self,x, distance_name = 'Euclidean', p=2, inv_covmat= None):
        
        md = -1 * np.ones((x.shape[0],x.shape[0]))

        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                if i == j:
                    dist_ij = 0
                    md[i,j] = dist_ij
                elif md[i,j] == -1:
                
                    dist_ij = self.get_distance(x[i,:], x[j,:], distance_name,p=p, inv_covmat= inv_covmat)
                    md[i,j] = dist_ij
                    md[j,i] = dist_ij
        return md


    def mahalanobis(self,v1,v2,inv_covmat):
        dif_vector = v1-v2
        return np.sqrt(np.matmul(np.matmul(dif_vector,inv_covmat),dif_vector.T))

    def minkowski(self,v1,v2, p=2):
        return np.power(np.sum(np.power(np.abs(v1-v2),p)),1./p)

    def cosine_distance(self,v1,v2):
        return 1-self.cosine_simil(v1,v2)

    def cosine_simil(self,v1,v2):
        ip = self.inner_p(v1,v2)
        norm_v1 = np.sqrt(self.inner_p(v1,v1))
        norm_v2 = np.sqrt(self.inner_p(v2,v2))
        simil = 0 if norm_v1 == 0 or norm_v2 == 0 else ip/(norm_v1*norm_v2)
        return simil

    def inner_p(self,v1,v2):
        return np.sum(np.multiply(v1,v2))
    
    def graph_distance_matrix(self,distances_matrix,distance, vertices_matrix = False):
        title = f'distance matrix'
        if vertices_matrix:
            path = f'Results/Distance to vertices/{distance.lower()}_vertices.png'
        else:
            path = f'Results/Distance Matrices/{distance.lower()}.png'

        self.graph_matrix(distances_matrix, title, path)
    
    def graph_matrix(self,distances, title, path):
        plt.figure()
        plt.imshow(distances, cmap='Blues_r')
        plt.title(title)
        plt.colorbar()
        plt.savefig(path)
        plt.close()
