import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from files.utils.constants import *
from files.utils.utility_functions import euclidean_distance, mad, plot


def find_neighbors(weights, pos, itself=False, diagonal=False):
    g, h = pos
    r, c = weights.shape[:2]
    neighbors = []
    for i in range(max([0, g-1]), min([r, g+2])):
        for j in range(max([0, h-1]), min([c, h+2])):
            if not diagonal and i != g and j != h:
                continue
            if i == g and j == h and not itself:
                continue
            neighbors.append(weights[i, j])
    return neighbors


def find_neighbors_idxs(weights, pos, itself=False, to_be_indexed=False, diagonal=False):
    g, h = pos
    r, c = weights.shape[:2]
    neighbors = []
    for i in range(max([0, g-1]), min([r, g+2])):
        for j in range(max([0, h-1]), min([c, h+2])):
            if not diagonal and i != g and j != h:
                continue
            if i == g and j == h and not itself:
                continue
            neighbors.append((i, j))
    if to_be_indexed:
        neighbors = np.array(neighbors)
        if len(neighbors.shape) == 1:
            return tuple([neighbors[0], neighbors[1]])
        return tuple([neighbors[:, 0], neighbors[:, 1]])
    return neighbors


class SelfOrganizingMaps:

    def __init__(self, n_rows, n_cols, data=None, lr=1, sigma=None, init_type=RND, n_dimensions=2, regularization=False, topology="rectangular"):
        sigma = (n_rows*n_cols)/10 if sigma is None else sigma
        self.sigma = sigma
        self.n_dimensions = n_dimensions
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.minisom = MiniSom(
            x=n_rows, y=n_cols, input_len=n_dimensions, learning_rate=lr, sigma=sigma, topology=topology)
        self.init_type = init_type
        if data is not None:
            self.initialize_weights(data)
        else:
            self.initialized = False

    def initialize_weights(self, data):
        self.initialized = True
        if self.init_type == PCA:
            self.minisom.pca_weights_init(data=data)
        elif self.init_type == RND:
            self.minisom.random_weights_init(data=data)

    def fit(self, epochs, data, verbose=False):
        if not self.initialized:
            self.initialize_weights(data)
        self.minisom.train(data, epochs, random_order=True, verbose=verbose)

    def predict(self, data):
        flat_net = self.minisom.get_weights()
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if len(data.shape) != 2:
            data = np.array([data])
        # for each point, take the closest weights to the point
        result = np.array([
            flat_net[self.minisom.winner(p)]
            for p in data
        ])
        return result

    def get_residuals(self, data):
        # To improve with np.vectorize
        return np.array([self.compute_outlier_factor(p) for p in data])

    def compute_outlier_factor(self, p):
        bmu = self.minisom.winner(p)
        neighbs = np.array(find_neighbors(
            self.minisom.get_weights(), bmu, itself=True))
        distances = np.linalg.norm([p for _ in neighbs] - neighbs, axis=1)
        return np.median(distances)

    def get_inliers(self, data, in_th=1.4826, indexes=False, v=NEW_INLIERS):
        if v == NEW_INLIERS:
            outlier_factors = np.array(
                [self.compute_outlier_factor(p) for p in data])
            # print(f"Residuals MAD: {residuals_mad}. Min of residuals: {np.min(residuals)}. Max of residuals: {np.max(residuals)}. Mean of residuals: {np.mean(residuals)}")
            if len(outlier_factors) > 0:
                outliers_mad = (np.median(outlier_factors) +
                                mad(outlier_factors))*in_th
                idxs = np.where(outlier_factors < outliers_mad)
                residuals = np.array([euclidean_distance(p, pr)
                                     for p, pr in zip(data, self.predict(data))])
                if len(residuals[idxs]) > 0:
                    idxs = np.where(residuals < np.median(residuals[idxs]))
                    if indexes:
                        return data[idxs], idxs
                    else:
                        return data[idxs]
            if indexes:
                return np.array([[np.inf, np.inf]]), np.array([np.inf])
            else:
                return np.array([[np.inf, np.inf]])
        else:
            residuals = np.array([euclidean_distance(p, pr)
                                 for p, pr in zip(data, self.predict(data))])
            if len(residuals) > 0:
                idxs = np.where(residuals < np.median(residuals)*in_th)[0]
                if indexes:
                    return data[idxs], idxs
                else:
                    return data[idxs]
            else:
                if indexes:
                    return np.array([[np.inf, np.inf]]), np.array([np.inf])
                else:
                    return np.array([[np.inf, np.inf]])

    def plot_weights_lattice(self, data=None, dpi=100, title="", ax=None):
        dict = {}
        weights = self.minisom.get_weights()
        for a, row in enumerate(weights):
            for b, weight in enumerate(row):
                neighb_idxs = find_neighbors_idxs(
                    weights, (a, b), to_be_indexed=True)
                dict[tuple(weight)] = weights[neighb_idxs]
        w = 1e-5  # Errorwidth
        h = 0.5   # Errorhead width

        if ax is None:
            fig, ax = plt.subplots(dpi=dpi)
            if title:
                fig.suptitle(title)
        if data is not None:
            plot(data, ax=ax, c='gray', alpha=0.4)
        ax.grid(False)

        for point, connections in dict.items():
            for outgoing in connections:
                dx = outgoing[0] - point[0]
                dy = outgoing[1] - point[1]
                ax.arrow(point[0], point[1],
                         dx, dy,
                         alpha=0.3,
                         width=w,
                         facecolor="k")

            ax.plot(*point,
                    marker="o", markersize=3,
                    markeredgecolor="k",
                    markerfacecolor="red")


if __name__ == "__main__":
    x = np.linspace(-4, 4, 200)
    y = np.sin(x)
    ds = np.dstack((x, y))[0]
    ds = np.vstack((ds, np.random.uniform(-5, 5, size=(100, 2))))
    som = SelfOrganizingMaps(3, 30, data=ds, sigma=0.9)
    som.fit(100, ds)
    inliers = som.get_inliers(ds)
    som.plot_weights_lattice(ds)
    plt.scatter(inliers[:, 0], inliers[:, 1])
    plt.show()
