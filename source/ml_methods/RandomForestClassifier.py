from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import RandomizedSearchCV

from source.ml_methods.SklearnMethod import SklearnMethod


class RandomForestClassifier(SklearnMethod):
    """
    Wrapper around the RandomForestClassifier scikit-learn's method.

    For usage and YAML specification, check :py:obj:`~source.ml_methods.SklearnMethod.SklearnMethod`.
    For valid parameters, see `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.

    If you are interested in plotting the coefficients of the random forest classifier model,
    consider running the :py:obj:`~source.reports.ml_reports.Coefficients.Coefficients` report.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_random_forest_classifier: # user-defined method name
            RandomForestClassifier: # name of the ML method
                # sklearn parameters (same names as in original sklearn class)
                n_estimators: 20 # number of trees in the forest
                random_state: 100 # controls the randomness of the boostrapping of examples while building trees
                # Additional parameter that determines whether to print convergence warnings
                show_warnings: True

    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        parameters = parameters if parameters is not None else {}

        if parameter_grid is not None:
            parameter_grid = parameter_grid
        else:
            parameter_grid = {"n_estimators": [10, 50, 100]}

        super(RandomForestClassifier, self).__init__(parameter_grid=parameter_grid, parameters=parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        params = self._parameters.copy()
        params["n_jobs"] = cores_for_training
        return RFC(**params)

    def can_predict_proba(self) -> bool:
        return True

    def get_params(self, label):
        params = self.models[label].estimator.get_params(deep=True) \
            if isinstance(self.models[label], RandomizedSearchCV) \
            else self.models[label].get_params(deep=True)
        params["feature_importances"] = self.models[label].feature_importances_.tolist()
        return params
