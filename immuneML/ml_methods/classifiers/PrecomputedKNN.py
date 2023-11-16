from sklearn.neighbors import KNeighborsClassifier


from immuneML.ml_methods.classifiers.SklearnMethod import SklearnMethod
from scripts.specification_util import update_docs_per_mapping


class PrecomputedKNN(SklearnMethod):
    """
    This is a wrapper of scikit-learn’s KNeighborsClassifier class.
    This ML method takes a pre-computed distance matrix, as created by the :ref:`Distance` or :ref:`CompAIRRDistance` encoders.
    If you would like to use a different encoding in combination with KNN, please use :ref:`KNN` instead.

    Please see the `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_
    of KNeighborsClassifier for the parameters.

    For usage instructions, check :py:obj:`~immuneML.ml_methods.classifiers.SklearnMethod.SklearnMethod`.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_knn_method:
            PrecomputedKNN:
                # sklearn parameters (same names as in original sklearn class)
                weights: uniform # always use this setting for weights
                n_neighbors: [5, 10, 15] # find the optimal number of neighbors
                # Additional parameter that determines whether to print convergence warnings
                show_warnings: True
            # if any of the parameters under KNN is a list and model_selection_cv is True,
            # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
            # and the optimal model will be selected
            model_selection_cv: True
            model_selection_n_folds: 5
        # alternative way to define ML method with default values:
        my_default_knn: PrecomputedKNN

    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        parameters = parameters if parameters is not None else {}
        parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(PrecomputedKNN, self).__init__(parameter_grid=parameter_grid, parameters=parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        params = self._parameters
        params["n_jobs"] = cores_for_training
        params["metric"] = "precomputed"
        return KNeighborsClassifier(**params)

    def get_params(self):
        return self.model.get_params(deep=True)

    def can_predict_proba(self) -> bool:
        return True

    def get_compatible_encoders(self):
        from immuneML.encodings.distance_encoding.CompAIRRDistanceEncoder import CompAIRRDistanceEncoder
        from immuneML.encodings.distance_encoding.DistanceEncoder import DistanceEncoder

        return [DistanceEncoder, CompAIRRDistanceEncoder]

    @staticmethod
    def get_documentation():
        doc = str(PrecomputedKNN.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.classifiers.SklearnMethod.SklearnMethod`.": SklearnMethod.get_usage_documentation("KNN"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc

