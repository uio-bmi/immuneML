from sklearn.decomposition import PCA as SklearnPCA
from immuneML.ml_methods.DimensionalityReduction import DimensionalityReduction


class PCA(DimensionalityReduction):

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnPCA(**self._parameters)
