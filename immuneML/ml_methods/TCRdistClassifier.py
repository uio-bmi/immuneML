import copy

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from immuneML.ml_methods.SklearnMethod import SklearnMethod
from immuneML.util.ParameterValidator import ParameterValidator


class TCRdistClassifier(SklearnMethod):
    """
    Implementation of a nearest neighbors classifier based on TCR distances as presented in
    Dash P, Fiore-Gartland AJ, Hertz T, et al. Quantifiable predictive features define epitope-specific T cell receptor repertoires.
    Nature. 2017; 547(7661):89-93. `doi:10.1038/nature22383 <https://www.nature.com/articles/nature22383>`_.

    This method is implemented using scikit-learn's KNeighborsClassifier with k determined at runtime from the training dataset size and weights
    linearly scaled to decrease with the distance of examples.

    Arguments:

        percentage (float): percentage of nearest neighbors to consider when determining receptor specificity based on known receptors (between 0 and 1)

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_tcr_method:
            TCRdistClassifier:
                percentage: 0.1
                show_warnings: True

    """

    def __init__(self, percentage: float, show_warnings: bool = True):
        super().__init__()

        ParameterValidator.assert_type_and_value(percentage, float, "TCRdistClassifier", "percentage", min_inclusive=0., max_inclusive=1.)

        self.percentage = percentage
        self.k = None
        self.label = None
        self.show_warnings = show_warnings

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        # compute k (number of nearest neighbors to consider) given the training dataset size (10% in the paper)
        self.k = int(X.shape[0] * self.percentage)

        # define function for computing weights which linearly decrease with distance
        def weights_func(distances):
            for point_dist_i, point_dist in enumerate(distances):
                if hasattr(point_dist, '__contains__') and 0. in point_dist:
                    distances[point_dist_i] = point_dist == 0.
                else:
                    distances[point_dist_i] = point_dist / np.sum(point_dist).astype(float)

        # make an object of KNN class with precomputed metric
        return KNeighborsClassifier(n_neighbors=self.k, weights=weights_func, metric='precomputed', n_jobs=cores_for_training)

    def get_params(self, label):
        return {**self.models[label].get_params(deep=True), **copy.deepcopy(vars(self))}

    def can_predict_proba(self) -> bool:
        return True

    def _get_model_filename(self):
        return "tcrdist_classifier"

