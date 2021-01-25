from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import RandomizedSearchCV

from immuneML.ml_methods.SklearnMethod import SklearnMethod
from scripts.specification_util import update_docs_per_mapping


class RandomForestClassifier(SklearnMethod):
    """
    This is a wrapper of scikit-learn’s RandomForestClassifier class. Please see the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
    of RandomForestClassifier for the parameters.

    Note: if you are interested in plotting the coefficients of the random forest classifier model,
    consider running the :ref:`Coefficients` report.

    For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_random_forest_classifier: # user-defined method name
            RandomForestClassifier: # name of the ML method
                # sklearn parameters (same names as in original sklearn class)
                random_state: 100 # always use this value for random state
                n_estimators: [10, 50, 100] # find the optimal number of trees in the forest
                # Additional parameter that determines whether to print convergence warnings
                show_warnings: True
            # if any of the parameters under RandomForestClassifier is a list and model_selection_cv is True,
            # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
            # and the optimal model will be selected
            model_selection_cv: True
            model_selection_n_folds: 5
        # alternative way to define ML method with default values:
        my_default_random_forest: RandomForestClassifier

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

    @staticmethod
    def get_documentation():
        doc = str(RandomForestClassifier.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": SklearnMethod.get_usage_documentation("RandomForestClassifier"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc
