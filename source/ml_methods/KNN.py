from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

from source.ml_methods.SklearnMethod import SklearnMethod


class KNN(SklearnMethod):
    """
    KNN wrapper of the KNeighborsClassifier scikit-learn's method.

    For usage and specification, check :py:obj:`~source.ml_methods.SklearnMethod.SklearnMethod`.
    For valid parameters, see `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_knn_method:
            KNN:
                # sklearn parameters (same names as in original sklearn class)
                n_neighbors: 5
                # Additional parameter that determines whether to print convergence warnings
                show_warnings: True

    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        parameters = parameters if parameters is not None else {}
        parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(KNN, self).__init__(parameter_grid=parameter_grid, parameters=parameters)


    def _get_ml_model(self, cores_for_training: int=2, X=None):
        params = self._parameters
        params["n_jobs"] = cores_for_training
        return KNeighborsClassifier(**params)

    def get_params(self, label):
        if isinstance(self.models[label], RandomizedSearchCV):
            params = self.models[label].estimator.get_params(deep=True)
        else:
            params = self.models[label].get_params(deep=True)
        return params

    def can_predict_proba(self) -> bool:
        return True
