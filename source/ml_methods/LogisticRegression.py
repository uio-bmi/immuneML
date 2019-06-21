from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV

from source.ml_methods.SklearnMethod import SklearnMethod


class LogisticRegression(SklearnMethod):

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):

        super(LogisticRegression, self).__init__()

        self._parameters = parameters if parameters is not None else {"max_iter": 5000}

        if parameter_grid is not None:
            self._parameter_grid = parameter_grid
        else:
            self._parameter_grid = {"max_iter": [10000],
                                    "penalty": ["l1"],
                                    "class_weight": ["balanced"]}

    def _get_ml_model(self, cores_for_training: int = 2):
        default = {"loss": "log", "n_jobs": cores_for_training}  # log loss + SGD classifier -> LR
        params = {**self._parameters, **default}
        return SGDClassifier(**params)

    def _can_predict_proba(self) -> bool:
        return True

    def get_params(self, label):
        params = self._models[label].estimator.get_params(deep=True) \
            if isinstance(self._models[label], RandomizedSearchCV) \
            else self._models[label].get_params(deep=True)
        params["coefficients"] = self._models[label].coef_.tolist()
        params["intercept"] = self._models[label].intercept_.tolist()
        return params
