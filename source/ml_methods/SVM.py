from sklearn.linear_model import SGDClassifier

from source.ml_methods.SklearnMethod import SklearnMethod


class SVM(SklearnMethod):
    # TODO: check the use case and add online learning method

    """
    Implements linear SVM using sklearn SGDClassifier class:
    SGD allows for online learning in case of large datasets

    Notes:
        - regularization term is called alpha with SGDClassifier
        - n_iter has to be set to a larger number (e.g. 1000) for the SGDClassifier to achieve the same performance
            as the original implementation of the algorithm
    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(SVM, self).__init__()

        self._parameters = parameters if parameters is not None else {"max_iter": 10000}

        if parameter_grid is not None:
            self._parameter_grid = parameter_grid
        else:
            self._parameter_grid = {"max_iter": [150000],
                                    "penalty": ["l1"],
                                    "class_weight": ["balanced"]}

    def _get_ml_model(self, cores_for_training: int = 2):
        default = {"loss": "hinge", "n_jobs": cores_for_training}  # hinge loss + SGD classifier -> SVM
        params = {**self._parameters, **default}
        return SGDClassifier(**params)

    def _can_predict_proba(self) -> bool:
        return False

    def get_params(self, label):
        params = self._models[label].get_params()
        params["coefficients"] = self._models[label].coef_
        params["intercept"] = self._models[label].intercept_
        return params
