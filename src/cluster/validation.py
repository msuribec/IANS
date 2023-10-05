
import numpy as np

from distances import DistanceMatrices
from itertools import product
from itertools import product
import numpy as np

class Validation:
    def __init__(self, memberships, data, distance_args, centroids = None, predictions = None):
        self.memberships = memberships
        self.data = data
        self.distance_args = distance_args
        self.N,self.M = self.data.shape
        self.K, N2 = self.memberships.shape

        self.n_points_per_cluster = None
        self.d_x_centers = None
        self.d_centers = None
        self.BH = None
        self.CH = None
        self.Hartigan = None
        self.xu = None
        self.silhouette = None
        self.DB = None

        self.SSB = None
        self.SSW = None
        if centroids is None:
            self.find_centroids()
        else:
            self.centroids = centroids
        
        if predictions is None:
            self.predictions = np.argmax(self.memberships, axis=0)

        else:
            self.predictions = predictions


        assert self.N == N2, "Number of data points and memberships do not match"

    def get_distance_to_centers(self):
        self.d_x_centers = DistanceMatrices().compute_distance_matrix_fast(self.centroids, self.data, **self.distance_args)
    
    def get_distance_center_to_centers(self):
        self.d_centers = DistanceMatrices().compute_distance_matrix_fast(self.centroids, self.centroids, **self.distance_args)

    def get_ssw(self):

        if self.d_x_centers is None:
            self.get_distance_to_centers()

        self.SSW = np.sum(np.multiply(self.memberships, self.d_x_centers**2))
    
    def get_ssb(self):

        mean_data = np.mean(self.data, axis = 0)
        d_centers_median = DistanceMatrices().compute_distance_vector(self.centroids, mean_data, **self.distance_args)

        if self.n_points_per_cluster is None:
            self.get_number_of_points_clusters()

        self.SSB = np.dot(d_centers_median**2,self.n_points_per_cluster)
    
    def get_CH_index(self):

        if self.SSB is None:
            self.get_ssb()
        if self.SSW is None:
            self.get_ssw()

        numerator = self.SSB/(self.K-1)
        denominator = self.SSW/(self.N-self.K)
        self.CH = numerator/denominator

    def get_BH_index(self):

        if self.SSW is None:
                    self.get_ssw()
        self.BH = self.SSW/self.K
    
    def get_Hartigan(self):
        if self.SSB is None:
            self.get_ssb()
        if self.SSW is None:
            self.get_ssw()
        self.Hartigan = np.log(self.SSB/self.SSW)
    
    def get_xu(self):
        if self.SSB is None:
            self.get_ssb()
        if self.SSW is None:
            self.get_ssw()
        self.xu = self.M * np.log(np.sqrt(self.SSW/self.M*self.N**2)) + np.log(self.K)

    def get_DB(self):
        if self.d_x_centers is None:
            self.get_distance_to_centers()
        if self.d_centers is None:
            self.get_distance_center_to_centers()
        distance_to_centers_in_cluster = np.multiply(self.d_x_centers, self.memberships)
        distance_to_centers_in_cluster[distance_to_centers_in_cluster == 0] = np.nan
        mean_dist_to_center = np.nanmean(distance_to_centers_in_cluster, axis=1)

        sum_db = 0
        for i in range(self.K):
            max_div = 0
            for j in range(self.K):
                if i != j:
                    sums = mean_dist_to_center[i] + mean_dist_to_center[j]
                    dij = self.d_centers[i,j]
                    div = (sums)/dij
                    if div > max_div:
                        max_div = div
            sum_db += max_div
        self.DB = sum_db/self.K

    def get_silhouette(self):
        if self.n_points_per_cluster is None:
            self.get_number_of_points_clusters()
        distance_points = DistanceMatrices().compute_distance_matrix_fast(self.data,self.data, **self.distance_args)
        distances_clusters = self.memberships @ distance_points
        sum_sx = 0
        for x in range(self.N):
            max_memb = np.max(self.memberships[:,x])
            x_clusters = np.where(self.memberships[:,x] == max_memb)[0] # Indexes of clusters where x belongs
            n_points_cluster = np.sum(np.max(self.memberships[x_clusters,:], axis = 0)) - 1 # Number of points that belong to the same cluster(s) as x
            if n_points_cluster == 0:
                sx = 0 # x is the only point in the cluster
            else:
                a_x = np.sum(distances_clusters[x_clusters,x])/n_points_cluster
                min_dist_to_other_clusters = np.inf
                index_min_dist_to_other_clusters = None
                for c in range(self.K):
                    if c not in x_clusters:
                        dist_to_points_in_cluster = distances_clusters[c,x]
                        if dist_to_points_in_cluster < min_dist_to_other_clusters:
                            min_dist_to_other_clusters = dist_to_points_in_cluster
                            index_min_dist_to_other_clusters = c
                if min_dist_to_other_clusters == np.inf:
                    sx = 0 # x belongs to all clusters
                else:
                    b_x = min_dist_to_other_clusters/self.n_points_per_cluster[index_min_dist_to_other_clusters]
                    sx = (b_x - a_x)/max(a_x,b_x)

            sum_sx += sx
        self.silhouette = sum_sx/self.N
        return self.silhouette

    def get_number_of_points_clusters(self):
        self.n_points_per_cluster = np.sum(self.memberships, axis = 1)
    
    def find_centroids(self):
        if self.n_points_per_cluster is None:
            self.get_number_of_points_clusters()
        centroids = (self.memberships @ self.data)/self.n_points_per_cluster.reshape(-1,1)
        self.centroids = centroids



    def get_all_internal_indexes(self):


        if self.BH is None:
            self.get_BH_index()
        if self.CH is None:
            self.get_CH_index()
        if self.Hartigan is None:
            self.get_Hartigan()
        if self.xu is None:
            self.get_xu()
        if self.DB is None:
            self.get_DB()
        if self.silhouette is None:
            self.get_silhouette()

        internal_indexes = {
            "CH": self.CH,
            "BH": self.BH,
            "Hartigan": self.Hartigan,
            "xu": self.xu,
            "DB": self.DB,
            "S": self.silhouette
        }

        return internal_indexes
    
    def get_all_external_indexes(self, labels):

        N = len(labels)
        combinations = set([(min(a,b),max(a,b)) for (a,b) in product(range(N), range(N)) if a!=b])

        # len(combinations) = N(N-1)/2

        TP = 0 # A the number of pairs of elements that are in the same subset in the labels and in the same subset in the predictions
        FN = 0 # D the number of pairs of elements in that are in diffetent subsets in the labels and in the same subset in the predictions
        FP = 0 # C the number of pairs of elements in that are in the same subset in the labels and in different subsets in the predictions
        TN = 0 # B: the number of pairs of elements that are in different subsets in the labels and in different subsets in the predictions

        for i, j in combinations:
            if i != j:
                if labels[i] == labels[j]:
                    if self.predictions[i] == self.predictions[j]:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if self.predictions[i] == self.predictions[j]:
                        FN += 1
                    else:
                        TN += 1
        
        prec =  TP / (TP + FP)
        recall = TP / (TP + FN)

        self.rand_index = (TP + TN) / len(combinations)
        self.fowlkes_mallows_index = np.sqrt(prec * recall)
        self.jaccard_index = TP / (TP + FP + FN)

        external_indexes = {
            "Rand": self.rand_index,
            "Fowlkes-Mallows": self.fowlkes_mallows_index,
            "Jaccard": self.jaccard_index
        }
        return external_indexes
