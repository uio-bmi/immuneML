from sklearn.cluster import Birch as SklearnBIRCH

from immuneML.ml_methods.Clustering.Clustering import Clustering


class BIRCH(Clustering):
    """
    This is a wrapper of scikit-learn’s BIRCH class for clustering. Please see the
    scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html>
    of BIRCH for the parameters.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_birch: # user-defined method name
            BIRCH: # name of the Clustering method
                # sklearn parameters (same names as in original sklearn class)
                threshold: 0.5 # The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold. Otherwise a new subcluster is started.
        my_default_birch: BIRCH
    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"n_clusters": 2}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(BIRCH, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnBIRCH(**self._parameters)
