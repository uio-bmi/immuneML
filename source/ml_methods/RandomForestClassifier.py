from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import RandomizedSearchCV

from source.ml_methods.SklearnMethod import SklearnMethod


class RandomForestClassifier(SklearnMethod):

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(RandomForestClassifier, self).__init__()

        self._parameters = parameters if parameters is not None else {}

        if parameter_grid is not None:
            self._parameter_grid = parameter_grid
        else:
            self._parameter_grid = {"n_estimators": [10, 50, 100]}

    def _get_ml_model(self, cores_for_training: int = 2):
        default = {"n_jobs": cores_for_training}
        params = {**self._parameters, **default}
        return RFC(**params)

    def _can_predict_proba(self) -> bool:
        return True

    def get_params(self, label):
        params = self._models[label].estimator.get_params(deep=True) \
            if isinstance(self._models[label], RandomizedSearchCV) \
            else self._models[label].get_params(deep=True)
        params["feature_importances"] = self._models[label].feature_importances_.tolist()
        return params
