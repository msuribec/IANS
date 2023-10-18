import umap
import umap.plot

class UMAP_EMBEDDING:
    def __init__(self, X_norm, params):
        self.X_norm = X_norm
        self.params = params

    def get_embedded_data(self, SEED=42):
        return umap.UMAP(**self.params, random_state= SEED).fit_transform(self.X_norm)

    def plot_umap(self, labels, SEED=42):
        mapper = umap.UMAP(**self.params, random_state= SEED).fit(self.X_norm)
        umap.plot.points(mapper, labels=labels)
