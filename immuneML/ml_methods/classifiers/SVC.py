from sklearn.svm import LinearSVC as SklearnSVC

from immuneML.ml_methods.classifiers.SklearnMethod import SklearnMethod
from scripts.specification_util import update_docs_per_mapping


class SVC(SklearnMethod):
    """
    This is a wrapper of scikit-learn’s LinearSVC class. Please see the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_
    of SVC for the parameters.

    Note: if you are interested in plotting the coefficients of the SVC model,
    consider running the :ref:`Coefficients` report.

    For usage instructions, check :py:obj:`~immuneML.ml_methods.classifiers.SklearnMethod.SklearnMethod`.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_svc: # user-defined method name
            SVC: # name of the ML method
                # sklearn parameters (same names as in original sklearn class)
                C: [0.01, 0.1, 1, 10, 100] # find the optimal value for C
                # Additional parameter that determines whether to print convergence warnings
                show_warnings: True
            # if any of the parameters under SVC is a list and model_selection_cv is True,
            # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
            # and the optimal model will be selected
            model_selection_cv: True
            model_selection_n_folds: 5
        # alternative way to define ML method with default values:
        my_default_svc: SVC

    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameter_grid = parameter_grid if parameter_grid is not None else {}
        _parameters = parameters if parameters is not None else {}

        super(SVC, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnSVC(**self._parameters)

    def can_predict_proba(self) -> bool:
        return False

    def get_params(self):
        params = self.model.get_params()
        params["coefficients"] = self.model.coef_[0].tolist()
        params["intercept"] = self.model.intercept_.tolist()
        return params

    @staticmethod
    def get_documentation():
        doc = str(SVC.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.classifiers.SklearnMethod.SklearnMethod`.": SklearnMethod.get_usage_documentation("SVC"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc
