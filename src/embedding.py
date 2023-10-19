import umap
import umap.plot
import matplotlib.pyplot as plt

class UMAP_EMBEDDING:
    def __init__(self, X_norm, params):
        self.X_norm = X_norm
        self.params = params

    def get_embedded_data(self, SEED=42):
        return umap.UMAP(**self.params, random_state= SEED).fit_transform(self.X_norm)

    def plot_umap(self, labels,path, SEED=42):
        mapper = umap.UMAP(**self.params, random_state= SEED).fit(self.X_norm)
        umap.plot.points(mapper, labels=labels,  color_key_cmap='Paired', background='white')
        plt.savefig(path)
        plt.close()
