import umap
import umap.plot
import matplotlib.pyplot as plt

class UMAP_EMBEDDING:
    """Class that embeds data using the UMAP algorithm
    Parameters:
        X_norm (numpy.ndarray):
            Data to embed
        params (dict):
            Dictionary with the arguments of the UMAP algorithm
    """
    def __init__(self, X_norm, params):
        self.X_norm = X_norm
        self.params = params

    def get_embedded_data(self, SEED=42):
        """Returns the embedded data
        Parameters:
            SEED (int):
                Seed for the random number generator
        Returns:
            numpy.ndarray:
                Embedded data
        """

        return umap.UMAP(**self.params, random_state= SEED).fit_transform(self.X_norm)

    def plot_umap(self, labels,path, SEED=42):
        """Plots the embedded data
        Parameters:
            labels (numpy.ndarray):
                Labels of the data
            path (str):
                Path where the plot will be saved
            SEED (int):
                Seed for the random number generator
        """
        mapper = umap.UMAP(**self.params, random_state= SEED).fit(self.X_norm)
        umap.plot.points(mapper, labels=labels,  color_key_cmap='Paired', background='white')
        plt.savefig(path)
        plt.close()
