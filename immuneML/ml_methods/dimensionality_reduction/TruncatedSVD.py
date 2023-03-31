from sklearn.decomposition import TruncatedSVD as SklearnTruncatedSVD
from immuneML.ml_methods.DimensionalityReduction import DimensionalityReduction



class TruncatedSVD(DimensionalityReduction):

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnTruncatedSVD(**self._parameters)
