import numpy as np
from numpy.linalg import inv
class CleanData:
    def __init__(self, data):
        self.data = data
        self.normalize()
        self.get_inv_covmat()

    def normalize(self):
        """Function to normalize the data"""
        self.norm_data = (self.data - np.min(self.data, axis=0)) / (np.max(self.data, axis=0) - np.min(self.data, axis=0))
        # self.norm_data = self.data/ np.max(self.data, axis=0)

    def get_inv_covmat(self):
        cov = np.cov(self.norm_data.T)    
        self.inv_covmat = inv(cov)

