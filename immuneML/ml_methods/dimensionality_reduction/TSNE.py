from sklearn.manifold import TSNE as SklearnTSNE
from immuneML.ml_methods.DimensionalityReduction import DimensionalityReduction

class TSNE(DimensionalityReduction):

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnTSNE(**self._parameters)
