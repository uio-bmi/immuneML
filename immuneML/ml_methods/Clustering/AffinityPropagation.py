from sklearn.cluster import AffinityPropagation as SklearnAffinityPropagation

from immuneML.ml_methods.Clustering.Clustering import Clustering


class AffinityPropagation(Clustering):
    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"damping": 0.5}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(AffinityPropagation, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnAffinityPropagation(**self._parameters)
