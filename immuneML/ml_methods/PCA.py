from sklearn.decomposition import PCA as SklearnPCA
from immuneML.ml_methods.DimensionalityReduction import DimensionalityReduction


class PCA(DimensionalityReduction):
    @classmethod
    def build_object(cls, **kwargs):
        return PCA(parameters=kwargs)

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(PCA, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnPCA(**self._parameters)
