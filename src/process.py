import numpy as np
from utils import ReadData

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



def process_data(file_name):

    reader = ReadData()
    
    data_df = reader.read_file(file_name)
    X = data_df.drop(['target'], axis=1).to_numpy()
    N, M = X.shape
    target_values = data_df['target'].values
    unique_classes = np.unique(target_values)
    n_classes = len(unique_classes)
    Y = np.zeros(N, dtype=np.int32)
    for i,target in enumerate(target_values):
        for j in range(n_classes):
            if target == unique_classes[j]:
                Y[i] = j
    
    cd = CleanData(X)
    X_norm = cd.norm_data
    inv_covmat = cd.inv_covmat

    return X_norm, Y, inv_covmat