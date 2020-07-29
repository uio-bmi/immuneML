from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC

from source.ml_methods.SklearnMethod import SklearnMethod


class SVM(SklearnMethod):
    """
    SVM wrapper of the corresponding scikit-learn's LinearSVC method.

    For usage and specification, check :py:obj:`~source.ml_methods.SklearnMethod.SklearnMethod`.
    For valid parameters, see `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_svm: # user-defined method name
            SVM: # name of the ML method
                penalty: l1 # use l1 regularization
        # alternative way to define ML method with default values:
        my_default_svm: SVM

    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(SVM, self).__init__()

        self._parameters = parameters if parameters is not None else {"max_iter": 10000, "multi_class": "crammer_singer"}

        if parameter_grid is not None:
            self._parameter_grid = parameter_grid
        else:
            self._parameter_grid = {}

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        params = {**self._parameters, **{}}
        return LinearSVC(**params)

    def _can_predict_proba(self) -> bool:
        return False

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
