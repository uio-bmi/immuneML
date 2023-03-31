from sklearn.manifold import TSNE as SklearnTSNE
from immuneML.ml_methods.DimensionalityReduction import DimensionalityReduction


class TSNE(DimensionalityReduction):
    @classmethod
    def build_object(cls, **kwargs):
        return TSNE(parameters=kwargs)

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(TSNE, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnTSNE(**self._parameters)
