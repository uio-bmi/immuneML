from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import RandomizedSearchCV

from source.ml_methods.SklearnMethod import SklearnMethod


class RandomForestClassifier(SklearnMethod):
    """
    Random Forest wrapper of the corresponding scikit-learn's method.

    For usage and specification, check :py:obj:`~source.ml_methods.SklearnMethod.SklearnMethod`.
    For valid parameters, see `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_random_forest_classifier: # user-defined method name
            RandomForestClassifier: # name of the ML method
                n_estimators: 20 # number of trees in the forest
                random_state: 100 # controls the randomness of the boostrapping of examples while building trees

    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(RandomForestClassifier, self).__init__()

        self._parameters = parameters if parameters is not None else {}

        if parameter_grid is not None:
            self._parameter_grid = parameter_grid
        else:
            self._parameter_grid = {"n_estimators": [10, 50, 100]}

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return RFC(**self._parameters)

    def can_predict_proba(self) -> bool:
        return True

    def get_params(self, label):
        params = self.models[label].estimator.get_params(deep=True) \
            if isinstance(self.models[label], RandomizedSearchCV) \
            else self.models[label].get_params(deep=True)
        params["feature_importances"] = self.models[label].feature_importances_.tolist()
        return params
