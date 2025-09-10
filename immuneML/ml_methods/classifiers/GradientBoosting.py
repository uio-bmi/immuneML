import copy

from sklearn.ensemble import GradientBoostingClassifier

from immuneML.ml_methods.classifiers.SklearnMethod import SklearnMethod
from scripts.specification_util import update_docs_per_mapping


class GradientBoosting(SklearnMethod):
    """
    Gradient Boosting classifier which wraps scikit-learn's GradientBoostingClassifier.
    Input arguments for the method are the same as supported by scikit-learn (see `GradientBoostingClassifier scikit-learn documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_ for details).

    For usage instructions, check :py:obj:`~immuneML.ml_methods.classifiers.SklearnMethod.SklearnMethod`.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_gradient_boosting:
                    GradientBoosting:
                        # arguments as defined by scikit-learn
                        n_estimators: 100
                        learning_rate: 0.1
                        max_depth: 3
                        random_state: 42

        """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(GradientBoosting, self).__init__(parameter_grid=parameter_grid, parameters=parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        params = copy.deepcopy(self._parameters) if self._parameters is not None else {}
        return GradientBoostingClassifier(**params)

    def can_predict_proba(self) -> bool:
        return True

    def can_fit_with_example_weights(self) -> bool:
        return True

    def get_params(self, for_refitting=False):
        params = copy.deepcopy(self.model.get_params())
        if not for_refitting:
            params["feature_importances"] = self.model.feature_importances_.tolist()
        return params

    @staticmethod
    def get_documentation():
        doc = str(GradientBoosting.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.classifiers.SklearnMethod.SklearnMethod`.": SklearnMethod.get_usage_documentation(
                "GradientBoosting"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc