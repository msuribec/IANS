import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from process import process_data
import ast
import pandas as pd

class Autoencoder:
    """Class that represents an autoencoder
    Attributes:
        X_norm (numpy.ndarray):
            normalized array (N x M ) of the data to cluster. N is the number of points in the dataset and M is the number of features
        X_train (numpy.ndarray):
            normalized array (N x M ) of the data to train the autoencoder. N is the number of points in the dataset and M is the number of features
        X_test (numpy.ndarray):
            normalized array (N x M ) of the data to test the autoencoder. N is the number of points in the dataset and M is the number of features
        hidden_size (int):
            number of neurons in the hidden layer
        learning_rate (float):
            learning rate of the autoencoder
        input_features (tf.keras.Input):
            input layer of the autoencoder
        encoded (tf.keras.layers.Dense):
            hidden layer of the autoencoder
        decoded (tf.keras.layers.Dense):
            output layer of the autoencoder
        autoencoder (tf.keras.Model):
            autoencoder model
        encoder (tf.keras.Model):
            encoder model
        encoded_data (numpy.ndarray):
            encoded data
        mse_test (float):
            mean squared error of the test data
        history (tf.keras.callbacks.History):
            history of the training of the autoencoder
    """
    def __init__(self, X_norm, X_train, X_test, hidden_size, learning_rate=0.001):
        self.X_norm = X_norm
        self.X_train = X_train
        self.X_test = X_test
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        input_dim = X_train.shape[1]

        encoding_dim = hidden_size 
        self.input_features = tf.keras.Input(shape=(input_dim,))
        self.encoded = tf.keras.layers.Dense(encoding_dim, activation='linear')(self.input_features)
        self.decoded = tf.keras.layers.Dense(input_dim, activation='linear')(self.encoded)
        self.autoencoder = tf.keras.Model(self.input_features, self.decoded)

        self.autoencoder.compile(optimizer='adam', loss='mse')
        # print(self.autoencoder.summary())

        self.encoder = tf.keras.Model(self.input_features, self.encoded)


    def train(self, X_val, epochs=1000, batch_size=16, shuffle=True, validation_split=0.2, verbose = 0):
        """Trains the autoencoder
        Parameters:
            x_val (numpy.ndarray):
                validation data
            epochs (int):
                number of epochs to train the autoencoder
            batch_size (int):
                batch size to train the autoencoder
            shuffle (bool):
                whether to shuffle the data or not
            validation_split (float):
                percentage of the data to use as validation
            verbose (int):
                verbosity mode
        """
        # Get the history of the model to plot
        self.history = self.autoencoder.fit(self.X_train, self.X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        validation_data=(X_val, X_val),
                        verbose = verbose)
    
    def predict(self):
        """Predicts the encoded data and calculates the mean squared error of the test data"""
        self.encoded_data = self.encoder(self.X_norm)

        X_pred = self.autoencoder(self.X_test)
        mse_test = np.mean(np.square(X_pred - self.X_test))
        self.mse_test = mse_test

    def plot_history(self):
        """Plots the history of the autoencoder"""

        plt.plot(self.history.history['loss'], color='#FF7E79',linewidth=3, alpha=0.5)
        plt.plot(self.history.history['val_loss'], color='#007D66', linewidth=3, alpha=0.4)
        plt.title('Model train vs Validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='best')
        plt.show()


    def plot_clusters(self, X, Y, target_names, title, vtitle, colors = ['#A43F98', '#5358E0', '#DE0202']):
        """Plots the clusters of the encoded data
        Parameters:
            X (numpy.ndarray):
                encoded data
            Y (numpy.ndarray):
                labels of the data
            target_names (list):
                list of the names of the clusters
            title (str):
                title of the plot
            vtitle (str):
                title of the axis
            colors (list):
                list of colors to use in the plot
        """
        n_clusters = len(target_names)
        indexes = list(range(n_clusters))

        lw = 2
        plt.figure(figsize=(9,7))

        for color, i, target_name in zip(colors, indexes, target_names):
            plt.scatter(X[Y == i, 0], X[Y == i, 1], color=color, alpha=1., lw=lw, label=target_name)
    
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title(title)
        plt.xlabel(vtitle + "1")
        plt.ylabel(vtitle + "2")
        plt.show()
    

def find_best_autoencoder_loW(X_norm, M, X_train, X_val,  X_test):
    """Finds the best autoencoder for low dimension
    Parameters:
        X_norm (numpy.ndarray):
            normalized array (N x M ) of the data to cluster. N is the number of points in the dataset and M is the number of features
        M (int):
            number of features
        X_train (numpy.ndarray):
            normalized array (N x M ) of the data to train the autoencoder. N is the number of points in the dataset and M is the number of features
        X_val (numpy.ndarray):
            normalized array (N x M ) of the data to validate the autoencoder. N is the number of points in the dataset and M is the number of features
        X_test (numpy.ndarray):
            normalized array (N x M ) of the data to test the autoencoder. N is the number of points in the dataset and M is the number of features
    Returns:
        best_low_dimension_data (numpy.ndarray):
            encoded data of the best autoencoder
    """
    min_error = np.inf
    min_index = 1
    results_low_d = {}
    for i in range(1, M):
        a = Autoencoder(X_norm, X_train, X_test, i)
        a.train(X_val)
        a.predict()
        current_error = a.mse_test
        results_low_d[i] = {'error': current_error, 'encoded_data': a.encoded_data}
        if current_error < min_error:
            min_error = current_error
            min_index = i
        df = pd.DataFrame(a.encoded_data.numpy())
        df.to_csv(f'try_{i}.csv', index=False)

    best_low_dimension_data = results_low_d[min_index]['encoded_data']
    print(min_index, results_low_d[min_index]['error'])
    return best_low_dimension_data.numpy()

def find_best_autoencoder_high(X_norm, M, X_train, X_val,  X_test):
    """Finds the best autoencoder for high dimension
    Parameters:
        X_norm (numpy.ndarray):
            normalized array (N x M ) of the data to cluster. N is the number of points in the dataset and M is the number of features
        M (int):
            number of features
        X_train (numpy.ndarray):
            normalized array (N x M ) of the data to train the autoencoder. N is the number of points in the dataset and M is the number of features
        X_val (numpy.ndarray):
            normalized array (N x M ) of the data to validate the autoencoder. N is the number of points in the dataset and M is the number of features
        X_test (numpy.ndarray):
            normalized array (N x M ) of the data to test the autoencoder. N is the number of points in the dataset and M is the number of features
    Returns:
        best_high_dimension_data (numpy.ndarray):
            encoded data of the best autoencoder
    """
    min_error = np.inf
    min_index = 0
    results_high_d = {}
    err_dif = 1
    err_diff_threshold = 0.0001

    i = M+1
    last_error = 1

    while err_dif > err_diff_threshold:
        a = Autoencoder(X_norm, X_train, X_test, i)
        a.train(X_val)
        a.predict()
        current_error = a.mse_test
        results_high_d[i] = {'error': current_error, 'encoded_data': a.encoded_data}

        err_dif = abs(last_error - current_error)
        last_error = current_error
        if current_error < min_error:
            min_error = current_error
            min_index = i
        
        i += 1
    


    best_high_dimension_data = results_high_d[min_index]['encoded_data']
    print(min_index, results_high_d[min_index]['error'])
    return best_high_dimension_data.numpy()

def find_best_low_high_dimension_data(dimension, X_norm, RANDOM_SEED = 42):
    """Finds the best autoencoder for low or high dimension
    Parameters:
        dimension (str):
            'low' or 'high'
        X_norm (numpy.ndarray):
            normalized array (N x M ) of the data to cluster. N is the number of points in the dataset and M is the number of features
        RANDOM_SEED (int):
            random seed
    Returns:
        best_low_high_dimension_data (numpy.ndarray):
            encoded data of the best autoencoder
    """

    X_train, X_tv = train_test_split(X_norm, test_size=0.4, random_state=RANDOM_SEED)
    X_test, X_val = train_test_split(X_tv, test_size=0.5, random_state=RANDOM_SEED)

    M = X_norm.shape[1]

    if dimension == 'low':
        return find_best_autoencoder_loW(X_norm,M, X_train, X_val,  X_test)
    elif dimension == 'high':
        return find_best_autoencoder_high(X_norm,M, X_train, X_val,  X_test)



if __name__ == '__main__':

    tf.keras.utils.set_random_seed(42)
    file_name = sys.argv.pop(1)
    csv_file_save = sys.argv.pop(1)
    type= sys.argv.pop(1)

    X_norm, Y, inv_covmat = process_data(file_name)


    if type == 'low' or type == 'high':
        X_norm = find_best_low_high_dimension_data(type, X_norm)
        inv_covmat = np.linalg.inv(np.cov(X_norm, rowvar=False))


    df = pd.DataFrame(X_norm)

    if Y is not None:
        df['target'] = Y
    
    df.to_csv(csv_file_save, index=False)
