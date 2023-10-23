import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

class Autoencoder:
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
        
        # Get the history of the model to plot
        self.history = self.autoencoder.fit(self.X_train, self.X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        validation_data=(X_val, X_val),
                        verbose = verbose)
    
    def predict(self):
        self.encoded_data = self.encoder(self.X_norm)

        X_pred = self.autoencoder(self.X_test)
        mse_test = np.mean(np.square(X_pred - self.X_test))
        self.mse_test = mse_test

    def plot_history(self):
        plt.plot(self.history.history['loss'], color='#FF7E79',linewidth=3, alpha=0.5)
        plt.plot(self.history.history['val_loss'], color='#007D66', linewidth=3, alpha=0.4)
        plt.title('Model train vs Validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='best')
        plt.show()


    def plot_clusters(self, X, Y, target_names, title, vtitle, colors = ['#A43F98', '#5358E0', '#DE0202']):
        
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
    min_error = np.inf
    min_index = 1
    results_low_d = {}
    print(M)
    for i in range(1, M):
        a = Autoencoder(X_norm, X_train, X_test, i)
        a.train(X_val)
        a.predict()
        current_error = a.mse_test
        results_low_d[i] = {'error': current_error, 'encoded_data': a.encoded_data}
        if current_error < min_error:
            min_error = current_error
            min_index = i

    best_low_dimension_data = results_low_d[min_index]['encoded_data']
    print(min_index, results_low_d[min_index]['error'])
    return best_low_dimension_data.numpy()

def find_best_autoencoder_high(X_norm, M, X_train, X_val,  X_test):
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

    X_train, X_tv = train_test_split(X_norm, test_size=0.4, random_state=RANDOM_SEED)
    X_test, X_val = train_test_split(X_tv, test_size=0.5, random_state=RANDOM_SEED)

    M = X_norm.shape[1]

    if dimension == 'low':
        return find_best_autoencoder_loW(X_norm,M, X_train, X_val,  X_test)
    elif dimension == 'high':
        return find_best_autoencoder_high(X_norm,M, X_train, X_val,  X_test)
