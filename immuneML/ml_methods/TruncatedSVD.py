from sklearn.decomposition import TruncatedSVD as SklearnTruncatedSVD
from immuneML.ml_methods.DimensionalityReduction import DimensionalityReduction


class TruncatedSVD(DimensionalityReduction):
    @classmethod
    def build_object(cls, **kwargs):
        return TruncatedSVD(parameters=kwargs)

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(TruncatedSVD, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnTruncatedSVD(**self._parameters)
