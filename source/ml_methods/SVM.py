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

    def __init__(self, parameter_grid: dict = None):
        super(SVM, self).__init__()

        if parameter_grid is not None:
            self._parameter_grid = parameter_grid
        else:
            self._parameter_grid = {"max_iter": [150000],
                                    "penalty": ["l1"],
                                    "class_weight": ["balanced", None]}

    def _get_ml_model(self, cores_for_training: int = 2):
        return SGDClassifier(loss="hinge", n_jobs=cores_for_training, tol=1e-3)  # hinge loss + SGD classifier -> SVM
