from abc import ABC

from immuneML.ml_methods.UnsupervisedSklearnMethod import UnsupervisedSklearnMethod


class Clustering(UnsupervisedSklearnMethod, ABC):
    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super().__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def get_params(self):
        params = self.model.get_params()
        return params

    def _calculate_auto_eps(self, X, cores_for_training, min_samples):
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        from scipy.sparse import csr_matrix
        from kneed import KneeLocator

        if "metric" in self._parameters:
            if self._parameters["metric"] == "precomputed":
                X = csr_matrix(X)
            neighbors = NearestNeighbors(n_neighbors=min_samples, metric=self._parameters["metric"], n_jobs=cores_for_training)
        else:
            neighbors = NearestNeighbors(n_neighbors=min_samples, metric="euclidean", n_jobs=cores_for_training)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)

        # Sort the distances by ascending order
        distances = np.sort(distances, axis=0)
        average_distances = distances[:, 1]

        import matplotlib.pyplot as plt
        # Plot the k-distance graph
        plt.plot(average_distances)

        S = 3
        if "metric" in self._parameters and self._parameters["metric"] == "precomputed":
            S = 200
        if "S" in self._parameters:
            S = self._parameters["S"]

        kl = KneeLocator(range(len(average_distances)), average_distances, S=S, curve='convex', direction='increasing', online=False)
        elbow = kl.elbow

        # Plot the elbow point
        plt.plot(elbow, average_distances[elbow], marker='o', markersize=10, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')

        plt.show()
        return float(average_distances[elbow])

    def get_compatible_encoders(self):
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
        from immuneML.encodings.word2vec.W2VSequenceEncoder import W2VSequenceEncoder
        from immuneML.encodings.distance_encoding.TCRdistEncoder import TCRdistEncoder

        return [KmerFrequencyEncoder, OneHotEncoder, W2VSequenceEncoder, TCRdistEncoder]
