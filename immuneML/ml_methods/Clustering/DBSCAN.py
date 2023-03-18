from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix

from immuneML.ml_methods.Clustering.Clustering import Clustering


class DBSCAN(Clustering):
    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"eps": 0.5}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(DBSCAN, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        if self._parameters["eps"] == "auto":
            self._calculate_optimal_eps(X, cores_for_training)

        if "S" in self._parameters:
            self._parameters.pop("S")
        return SklearnDBSCAN(**self._parameters)

    def _calculate_optimal_eps(self, X, cores_for_training):
        from kneed import KneeLocator
        if "metric" in self._parameters:
            if self._parameters["metric"] == "precomputed":
                X = csr_matrix(X)
            neighbors = NearestNeighbors(n_neighbors=self._parameters["min_samples"], metric=self._parameters["metric"], n_jobs=cores_for_training)
        else:
            neighbors = NearestNeighbors(n_neighbors=self._parameters["min_samples"], metric="euclidean", n_jobs=cores_for_training)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)

        # Sort the distances by ascending order
        distances = np.sort(distances, axis=0)
        average_distances = distances[:, 1]

        import matplotlib.pyplot as plt
        # Plot the k-distance graph
        plt.plot(average_distances)

        kl = KneeLocator(range(len(average_distances)), average_distances, S=self._parameters["S"], curve='convex', direction='increasing', online=True)
        elbow = kl.elbow

        # Plot the elbow point
        plt.plot(elbow, average_distances[elbow], marker='o', markersize=10, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')

        plt.show()
        self._parameters["eps"] = average_distances[elbow]
