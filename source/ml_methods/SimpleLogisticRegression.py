from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from source.ml_methods.SklearnMethod import SklearnMethod


class SimpleLogisticRegression(SklearnMethod):
    """
    Logistic regression wrapper of the corresponding scikit-learn's method.

    For usage and specification, check :py:obj:`~source.ml_methods.SklearnMethod.SklearnMethod`.
    For valid parameters, see `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_logistic_regression: # user-defined method name
            SimpleLogisticRegression: # name of the ML method
                penalty: l1 # use l1 regularization
                C: 10 # regularization constant
        # alternative way to define ML method with default values:
        my_default_logistic_regression: SimpleLogisticRegression

    """

    default_parameters = {"max_iter": 1000, "solver": "saga"}

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(SimpleLogisticRegression, self).__init__()

        self._parameters = {**self.default_parameters, **(parameters if parameters is not None else {})}

        if parameter_grid is not None:
            self._parameter_grid = parameter_grid
        else:
            self._parameter_grid = {"max_iter": [1000],
                                    "penalty": ["l1", "l2"],
                                    "C": [0.001, 0.01, 0.1, 10, 100, 1000],
                                    "class_weight": ["balanced"]}

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return LogisticRegression(**self._parameters)

    def can_predict_proba(self) -> bool:
        return True

    def get_params(self, label):
        if label is None:
            tmp_label = list(self.models.keys())[0]
        else:
            tmp_label = label

        params = self.models[tmp_label].estimator.get_params() if isinstance(self.models[tmp_label], RandomizedSearchCV) \
            else self.models[tmp_label].get_params()
        params["coefficients"] = self.models[tmp_label].coef_[0].tolist()
        params["intercept"] = self.models[tmp_label].intercept_.tolist()
        return params
