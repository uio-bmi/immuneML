from sklearn.cluster import DBSCAN as SklearnDBSCAN

from immuneML.ml_methods.Clustering.Clustering import Clustering


class DBSCAN(Clustering):
    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"eps": 0.5}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(DBSCAN, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        if self._parameters["eps"] == "auto":
            self._calculate_auto_eps(X, cores_for_training)

        if "S" in self._parameters:
            self._parameters.pop("S")
        return SklearnDBSCAN(**self._parameters)
