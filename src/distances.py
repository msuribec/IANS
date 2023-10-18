from utils import create_folders_if_not_exist
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
 

class DistanceMatrices:
    """Class that represents a distance matrix
    """

    def __init__(self, main_path = 'Results'):
        self.main_path = main_path
        pass

    def compute_distance_matrices(self, data, distance_definitions, vertices = None):
        """Computes and graphs the distance matrices for the given data and distance definitions
        Parameters:
        data (np.array):
            array (N x M) of the normalized data to cluster. N is the number of points in the dataset and M is the number of features

        distance_definitions (dict):
            dictionary specifing the distances to compute the matrices. Each entry in the dictionary is a dictionary
            with the name of the distance and the parameters of the distance.
        
        For example:
        
        Given the following dictionary, this function will compute the Manhatta and Minkowski (with p = 3) distance matrices
        {
            'Manhattan':{
                'distance_name': 'Manhattan'
            },
            'Minkowski with p = 3': {
                'distance_name': 'Minkowski',
                'p' : 3
            }
        }

        The accepted distances are: Euclidean, Manhattan, Minkowski, Mahalanobis and the pseudo distance Cosine
        
        - The Euclidean, Manhattan and Cosine distances do not require extra parameters.
        - The Minkowski distance requires the parameter p
        - The Mahalanobis distance requires the inverse of the covariance matrix of the data
        
        Reminder: The Cosine distance is not a real distance, but a similarity measure. It is computed as 1 - cosine_similarity
        """
        create_folders_if_not_exist([self.main_path, f'{self.main_path}/Distance Matrices',f'{self.main_path}/Distance to vertices'])
        self.distance_matrices = {}
        self.distance_matrices_vertices = {}
        for distance_id in distance_definitions:
            params = distance_definitions[distance_id]
            dist_matrix = self.compute_distance_matrix_fast(data,data,**params)
            self.distance_matrices[distance_id] = dist_matrix
            self.graph_distance_matrix(dist_matrix, distance_id, False)
            if vertices is not None:
                dist_matrix_vertices = self.compute_distance_matrix_fast(data,vertices,**params)
                self.distance_matrices_vertices[distance_id] = dist_matrix_vertices
                self.graph_distance_matrix(dist_matrix_vertices, distance_id, True)

    def get_distance(self,v1,v2,distance_name= 'Euclidean', p=None, inv_covmat = None):
        """Computes the distance between two vectors
        Parameters:
            v1 (np.array):
                vector 1
            v2 (np.array):
                vector 2
            distance (str):
                name of the distance to compute. Accepted values are: Euclidean, Manhattan, Minkowski, Mahalanobis and Cosine
            p (int):
                parameter for the Minkowski distance. If distance is not Minkowski, this parameter is ignored
            inv_covmat (np.array):
                inverse of the covariance matrix of the data. If distance is not Mahalanobis, this parameter is ignored
        Returns:
            dist_v1v2 (float):
                distance between v1 and v2
        """
        if distance_name == 'Euclidean':
            dist_v1v2 = self.minkowski(v1,v2, 2)
        elif distance_name == 'Manhattan':
            dist_v1v2 =  self.minkowski(v1,v2, 1)
        elif distance_name == 'Minkowski':
            assert p is not None, "p must be specified when using p_distance"
            dist_v1v2 =  self.minkowski(v1,v2, p)
        elif distance_name == 'Mahalanobis':
            dist_v1v2 = self.mahalanobis(v1,v2,inv_covmat)   
        elif distance_name == 'Cosine':
            dist_v1v2 = self.cosine_distance(v1,v2)
        return dist_v1v2

    def get_distance_vector(self,V,v1,distance_name= 'Euclidean', p=None, inv_covmat = None):
        """Computes the distance between two vectors
        Parameters:
            v1 (np.array):
                vector 1
            v2 (np.array):
                vector 2
            distance (str):
                name of the distance to compute. Accepted values are: Euclidean, Manhattan, Minkowski, Mahalanobis and Cosine
            p (int):
                parameter for the Minkowski distance. If distance is not Minkowski, this parameter is ignored
            inv_covmat (np.array):
                inverse of the covariance matrix of the data. If distance is not Mahalanobis, this parameter is ignored
        Returns:
            dist_v1v2 (float):
                distance between v1 and v2
        """
        if distance_name == 'Euclidean':
            dist_v1v2 = np.sqrt(np.sum(np.power((V-v1),2), axis = 1))
        elif distance_name == 'Manhattan':
            dist_v1v2 = np.sum(np.abs(V-v1), axis = 1)
        elif distance_name == 'Minkowski':
            assert p is not None, "p must be specified when using p_distance"
            return np.sum((V-v1)**2, axis = 1)
        elif distance_name == 'Mahalanobis':
            # Memory efficient
            dist_v1v2 =  distance.cdist(V, v1.reshape(1,-1), 'mahalanobis', VI=inv_covmat)
            dist_v1v2 = dist_v1v2.reshape(V.shape[0],)
        elif distance_name == 'Cosine':
            norm_V = np.sqrt(np.sum(V**2, axis = 1))
            norm_v1 = np.sqrt(np.sum(v1**2))
            simil = np.dot(V,v1)/(norm_V*norm_v1)
            simil[np.isnan(simil)] = 0
            dist_v1v2 = 1 - simil
        return dist_v1v2
        
    def compute_distance_matrix(self,x, y = None, distance_name = 'Euclidean', p=2, inv_covmat= None):

        """Computes the distance matrix for the given data and distance definition
        Parameters:
            x (np.array):
                array (N1 x M) of the normalized data. N1 is the number of points in the dataset and M is the number of features
            y (np.array):
                array (N2 x M) of the normalized data. N1 is the number of points in the dataset and M is the number of features
            distance_name (str):
                name of the distance to compute. Accepted values are: Euclidean, Manhattan, Minkowski, Mahalanobis and Cosine
            p (int):
                parameter for the Minkowski distance. If distance is not Minkowski, this parameter is ignored
            inv_covmat (np.array):
                inverse of the covariance matrix of the data. If distance is not Mahalanobis, this parameter is ignored
        Returns:
            md (np.array):
                distance matrix (N1 X N2) where position i,j holds the distance between point i of matrix x and and point j of matrix y
        """

        symmetry = False

        if y is None:
            y = x
            symmetry = True

        md = -1 * np.ones((x.shape[0],y.shape[0]))

        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                if i == j and symmetry:
                    dist_ij = 0
                    md[i,j] = dist_ij
                elif md[i,j] == -1:
                    dist_ij = self.get_distance(x[i,:], y[j,:], distance_name,p=p, inv_covmat= inv_covmat)
                    md[i,j] = dist_ij
                    if y is symmetry:
                        md[j,i] = dist_ij
        return md
    
        
    def compute_distance_matrix_fast(self,x, y, distance_name = 'Euclidean', p=2, inv_covmat= None):

        """Computes the distance matrix for the given data and distance definition
        Parameters:
            x (np.array):
                array (N1 x M) of the normalized data. N1 is the number of points in the dataset and M is the number of features
            y (np.array):
                array (N2 x M) of the normalized data. N1 is the number of points in the dataset and M is the number of features
            distance_name (str):
                name of the distance to compute. Accepted values are: Euclidean, Manhattan, Minkowski, Mahalanobis and Cosine
            p (int):
                parameter for the Minkowski distance. If distance is not Minkowski, this parameter is ignored
            inv_covmat (np.array):
                inverse of the covariance matrix of the data. If distance is not Mahalanobis, this parameter is ignored
        Returns:
            md (np.array):
                distance matrix (N1 X N2) where position i,j holds the distance between point i of matrix x and and point j of matrix y
        """

        md = np.zeros((x.shape[0],y.shape[0]))
        for j in range(y.shape[0]):
            md[:,j] = self.get_distance_vector(x,y[j,:],distance_name,p=p, inv_covmat= inv_covmat)
        return md
    
    
    def compute_distance_vector(self,X, y, distance_name = 'Euclidean', p=2, inv_covmat= None):
        

        """Computes the distance vector between each vector of X and the vector y for the given data and distance definition
        Parameters:
            X (np.array):
                array (N1 x M) of the normalized data. N1 is the number of points in the dataset and M is the number of features
            y (np.array):
                array (M X 1) of the normalized data. M is the number of features
            distance_name (str):
                name of the distance to compute. Accepted values are: Euclidean, Manhattan, Minkowski, Mahalanobis and Cosine
            p (int):
                parameter for the Minkowski distance. If distance is not Minkowski, this parameter is ignored
            inv_covmat (np.array):
                inverse of the covariance matrix of the data. If distance is not Mahalanobis, this parameter is ignored
        Returns:
            md (np.array):
                distance matrix (N1 X N2) where position i,j holds the distance between point i of matrix x and and point j of matrix y
        """

        md = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            md[i] =  self.get_distance(X[i,:], y, distance_name,p=p, inv_covmat= inv_covmat)
                    
        return md


    def mahalanobis(self,v1,v2,inv_covmat):
        """Computes the Mahalanobis distance between two vectors
        Parameters:
            v1 (np.array):
                vector 1
            v2 (np.array):
                vector 2
            inv_covmat (np.array):
                inverse of the covariance matrix of the data
        Returns:
            dist_v1v2 (float):
                Mahalanobis distance between v1 and v2
        """
        dif_vector = v1-v2
        squared_d = np.matmul(np.matmul(dif_vector,inv_covmat),dif_vector.T)
        return np.sqrt(squared_d)

    def minkowski(self,v1,v2, p=2):
        """Computes the Minkowski distance between two vectors
        Parameters:
            v1 (np.array):
                vector 1
            v2 (np.array):
                vector 2
            p (int):
                parameter for the Minkowski distance
        """
        if p == 2:
            return np.sqrt(np.sum(np.power(v1-v2,p)))
        elif p == 1:
            return np.sqrt(np.sum(np.abs(v1-v2)))
        else:
            return np.power(np.sum(np.power(np.abs(v1-v2),p)),1./p)


    def cosine_distance(self,v1,v2):
        """Computes the cosine distance between two vectors
        Parameters:
            v1 (np.array):
                vector 1
            v2 (np.array):
                vector 2
        Returns:
            dist_v1v2 (float):
                cosine distance between v1 and v2. This is pseudo distance and is computed as 1 - cosine_similarity
        """
        return 1-self.cosine_simil(v1,v2)

    def cosine_simil(self,v1,v2):
        """Computes the cosine similarity between two vectors
        Parameters:
            v1 (np.array):
                vector 1
            v2 (np.array):
                vector 2
        Returns:
            simil (float):
                cosine similarity between v1 and v2.
                To calculate the inner product we use the inner_p function.
                It is possible to induce another norm (and therefore another distance) product by changing the inner_p function
        """
        ip = self.inner_p(v1,v2)
        norm_v1 = np.sqrt(self.inner_p(v1,v1))
        norm_v2 = np.sqrt(self.inner_p(v2,v2))
        simil = 0 if norm_v1 == 0 or norm_v2 == 0 else ip/(norm_v1*norm_v2)
        return simil

    def inner_p(self,v1,v2):
        """Computes the inner product between two vectors. This is the typical inner product in Rn
        Parameters:
            v1 (np.array):
                vector 1
            v2 (np.array):
                vector 2
        Returns:
            inner_p (float):
                inner product between v1 and v2
        """
        return np.sum(np.multiply(v1,v2))
    
    def graph_distance_matrix(self,distances_matrix,distance, vertices_matrix = False):
        """Graphs the distance matrix and saves it to the Results folder
        Parameters:
            distances_matrix (np.array):
                distance matrix to graph
            distance (str):
                name of the distance. This is used for the title of the graph and the name of the file
            vertices_matrix (bool):
                if True, the graph is saved to the Results/Distance to vertices folder. Otherwise, it is saved to the Results/Distance Matrices folder
        """
        title = f'distance matrix'
        if vertices_matrix:
            path = f'{self.main_path}/Distance to vertices/{distance.lower()}_vertices.png'
            title = f'distance from points to vertices'
        else:
            path = f'{self.main_path}/Distance Matrices/{distance.lower()}.png'

        self.graph_matrix(distances_matrix, title, path)
    
    def graph_matrix(self,distances, title, path):
        """Graphs the given matrix and saves it to the specified path
        Parameters:
            distances (np.array):
                matrix to graph
            title (str):
                title of the graph
            path (str):
                path where the graph will be saved
        """
        plt.figure()
        plt.imshow(distances, cmap='Blues_r')
        plt.title(title)
        plt.colorbar()
        plt.savefig(path)
        plt.close()
