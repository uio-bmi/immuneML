from abc import ABC

from immuneML.ml_methods.UnsupervisedSklearnMethod import UnsupervisedSklearnMethod
from immuneML.data_model.encoded_data.EncodedData import EncodedData

from scipy.sparse import csr_matrix


class DimensionalityReduction(UnsupervisedSklearnMethod, ABC):
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
        if type(self.model).__name__ in ["PCA"]:
            if isinstance(X, csr_matrix):
                X = X.toarray()
        encoded_data.set_dim_reduced_examples(self.model.fit_transform(X))

        return self.model
