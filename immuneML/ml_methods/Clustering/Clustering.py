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

    def _calculate_auto_eps(self, X, cores_for_training, min_samples, sensitivity: float = 3.0):
        if "metric" in self._parameters and self._parameters["metric"] == "precomputed":
            X = self._convert_to_csr(X)
        neighbors = self._create_nearest_neighbors(X, cores_for_training, min_samples)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)

        average_distances = self._calculate_average_distances(distances)
        S = self._adjust_sensitivity(sensitivity)

        elbow = self._find_elbow(average_distances, S)

        return float(average_distances[elbow])

    def _convert_to_csr(self, X):
        from scipy.sparse import csr_matrix
        return csr_matrix(X)

    def _create_nearest_neighbors(self, X, cores_for_training, min_samples):
        from sklearn.neighbors import NearestNeighbors

        if "metric" in self._parameters:
            return NearestNeighbors(n_neighbors=min_samples, metric=self._parameters["metric"], n_jobs=cores_for_training)
        else:
            return NearestNeighbors(n_neighbors=min_samples, metric="euclidean", n_jobs=cores_for_training)

    def _calculate_average_distances(self, distances):
        import numpy as np

        distances = np.sort(distances, axis=0)
        return distances[:, 1]

    def _adjust_sensitivity(self, sensitivity):
        if "metric" in self._parameters and self._parameters["metric"] == "precomputed":
            return 200
        return sensitivity

    def _find_elbow(self, average_distances, sensitivity):
        from kneed import KneeLocator

        kl = KneeLocator(range(len(average_distances)), average_distances, S=sensitivity, curve='convex', direction='increasing', online=False)
        return kl.elbow

    def get_compatible_encoders(self):
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
        from immuneML.encodings.word2vec.W2VSequenceEncoder import W2VSequenceEncoder
        from immuneML.encodings.distance_encoding.TCRdistEncoder import TCRdistEncoder

        return [KmerFrequencyEncoder, OneHotEncoder, W2VSequenceEncoder, TCRdistEncoder]
