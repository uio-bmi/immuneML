from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

from source.ml_methods.SklearnMethod import SklearnMethod


class KNN(SklearnMethod):
    """
    KNN wrapper of the KNeighborsClassifier scikit-learn's method.

    For usage and specification, check SklearnMethod class.
    For valid parameters, see: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(KNN, self).__init__()

        self._parameters = parameters if parameters is not None else {}
        self._parameter_grid = parameter_grid if parameter_grid is not None else {}

    def _get_ml_model(self, cores_for_training: int = 2):
        return KNeighborsClassifier(**self._parameters)

    def get_params(self, label):
        if isinstance(self.models[label], RandomizedSearchCV):
            params = self.models[label].estimator.get_params(deep=True)
        else:
            params = self.models[label].get_params(deep=True)
        return params
