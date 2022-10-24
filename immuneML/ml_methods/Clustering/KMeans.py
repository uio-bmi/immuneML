from sklearn.cluster import KMeans as SklearnKMeans

from immuneML.ml_methods.Clustering.Clustering import Clustering


class KMeans(Clustering):
    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"n_clusters": 2, "n_init": 10}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(KMeans, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnKMeans(**self._parameters)
