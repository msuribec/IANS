import numpy as np

class CleanData:
    """Class that represents a preprocessed dataset
    Attributes:
        data (np.array):
            array (N x M) of the data to cluster. N is the number of points in the dataset and M is the number of features
    """
    def __init__(self, data):
        self.data = data
        self.normalize()
        self.get_inv_covmat()


    def normalize(self):
        """Function to normalize the data"""
        self.norm_data = (self.data - np.min(self.data, axis=0)) / (np.max(self.data, axis=0) - np.min(self.data, axis=0))
        # self.norm_data = self.data/ np.max(self.data, axis=0)

    def get_inv_covmat(self):
        """Function to compute the inverse of the covariance matrix of the data"""  
        self.inv_covmat = np.linalg.inv(np.cov(self.norm_data, rowvar=False))
