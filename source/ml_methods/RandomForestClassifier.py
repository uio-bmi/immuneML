from sklearn.ensemble import RandomForestClassifier as RFC

from source.ml_methods.SklearnMethod import SklearnMethod


class RandomForestClassifier(SklearnMethod):

    def __init__(self, parameter_grid: dict = None):
        super(RandomForestClassifier, self).__init__()
        if parameter_grid is not None:
            self._parameter_grid = parameter_grid
        else:
            self._parameter_grid = {"n_estimators": [10, 50, 100]}

    def _get_ml_model(self, cores_for_training: int = 2):
        return RFC(n_jobs=cores_for_training)
