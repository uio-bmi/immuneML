from sklearn.linear_model import SGDClassifier
from source.ml_methods.SklearnMethod import SklearnMethod


class LogisticRegression(SklearnMethod):

    def __init__(self, parameter_grid: dict = None):

        super(LogisticRegression, self).__init__()

        if parameter_grid is not None:
            self._parameter_grid = parameter_grid
        else:
            self._parameter_grid = {"max_iter": [10000],
                                    "penalty": ["l1"],
                                    "class_weight": ["balanced"]}

    def _get_ml_model(self, cores_for_training: int = 2):
        return SGDClassifier(loss="log", n_jobs=cores_for_training, tol=1e-3)  # log loss + SGD classifier -> LR
