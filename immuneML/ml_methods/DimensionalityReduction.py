from abc import ABC

from immuneML.ml_methods.UnsupervisedSklearnMethod import UnsupervisedSklearnMethod
from immuneML.data_model.encoded_data.EncodedData import EncodedData

from scipy.sparse import csr_array
from scipy.sparse import csr_matrix


class DimensionalityReduction(UnsupervisedSklearnMethod, ABC):
    """
    This is an abstract base class for unsupervised machine learning methods used for dimensionality reduction. It wraps around scikit-learn's unsupervised methods.

    Note: This class should not be instantiated directly. Instead, it should be used as a base class for specific dimensionality reduction methods (e.g., PCA, t-SNE).

    The implemented classes should provide their own implementation of the _get_ml_model method, which returns an instance of the specific scikit-learn method.

    The fit_transform method is used to fit the model to the data and then transform the data to its lower-dimensional representation.

    Each implementing class should have a build_object class method that constructs an instance of the class with the given parameters.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_dimensionality_reduction: # user-defined method name
            PCA: # name of the ML method
                # sklearn parameters (same names as in original sklearn class)
                n_components: 2 # reduce to 2 dimensions
            # alternative way to define ML method with default values:
            my_default_dimensionality_reduction: PCA

    """
    def get_params(self):
        params = self.model.get_params()
        return params

    @classmethod
    def build_object(cls, **kwargs):
        return cls(parameters=kwargs)

    def __init__(self, parameters: dict = None):
        _parameters = parameters if parameters is not None else {}
        super().__init__(parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        raise NotImplementedError

    def fit_transform(self, encoded_data: EncodedData, cores_for_training: int = 2):
        X = encoded_data.examples

        self.model = self._get_ml_model(cores_for_training, X)
        if type(self.model).__name__ in ["PCA", "TSNE"]:
            if isinstance(X, csr_matrix):
                X = X.toarray()
        encoded_data.set_dim_reduced_examples(self.model.fit_transform(X))

        return self.model
