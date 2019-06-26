from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC

from source.ml_methods.SklearnMethod import SklearnMethod


class SVM(SklearnMethod):
    # TODO: check the use case and add online learning method

    """
    Implements linear SVM using sklearn LinearSVC class:
    LinearSVC allows for online learning in case of large datasets

    Notes:
        - regularization term is called alpha with LinearSVC
        - n_iter has to be set to a larger number (e.g. 1000) for the SGDClassifier to achieve the same performance
            as the original implementation of the algorithm
    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(SVM, self).__init__()

        self._parameters = parameters if parameters is not None else {"max_iter": 10000, "multi_class": "crammer_singer"}

        if parameter_grid is not None:
            self._parameter_grid = parameter_grid
        else:
            self._parameter_grid = {}

    def _get_ml_model(self, cores_for_training: int = 2):
        params = {**self._parameters, **{}}
        return LinearSVC(**params)

    def _can_predict_proba(self) -> bool:
        return False

    def get_params(self, label):
        params = self.models[label].estimator.get_params() if isinstance(self.models[label], RandomizedSearchCV) \
            else self.models[label].get_params()
        params["coefficients"] = self.models[label].coef_.tolist()
        params["intercept"] = self.models[label].intercept_.tolist()
        return params
