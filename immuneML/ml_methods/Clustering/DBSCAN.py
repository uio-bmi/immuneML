from sklearn.cluster import DBSCAN as SklearnDBSCAN

from immuneML.ml_methods.Clustering.Clustering import Clustering


class DBSCAN(Clustering):
    """
    This is a wrapper of scikit-learn’s DBSCAN class for clustering. Please see the
    scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>
    of DBSCAN for the parameters.

    Additionally, eps parameter can be set to "auto" for automatic eps calculation using kneed package.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_dbscan: # user-defined method name
            DBSCAN: # name of the Clustering method
                # sklearn parameters (same names as in original sklearn class)
                eps: 0.5 # The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        my_default_dbscan: DBSCAN
    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"eps": 0.5}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        self.auto_eps = False
        self.auto_eps_s = 3.0

        if "S" in _parameters:
            self.auto_eps_s = _parameters["S"]
            _parameters.pop("S")

        super(DBSCAN, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        if self.auto_eps:
            self._parameters["eps"] = "auto"
        else:
            self.auto_eps = self._parameters["eps"] == "auto"

        if self._parameters["eps"] == "auto":
            self._parameters["eps"] = self._calculate_auto_eps(X, cores_for_training, self._parameters["min_samples"], self.auto_eps_s)

        return SklearnDBSCAN(**self._parameters)
