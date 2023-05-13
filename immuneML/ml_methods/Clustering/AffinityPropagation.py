from sklearn.cluster import AffinityPropagation as SklearnAffinityPropagation

from immuneML.ml_methods.Clustering.Clustering import Clustering


class AffinityPropagation(Clustering):
    """
    This is a wrapper of scikit-learn’s AffinityPropagation class for clustering. Please see the
    scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html>
    of AffinityPropagation for the parameters.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

    my_affinity_propagation: # user-defined method name
        AffinityPropagation: # name of the Clustering method
            # sklearn parameters (same names as in original sklearn class)
            damping: 0.5 # Damping factor (between 0.5 and 1)
    my_default_affinity_propagation: AffinityPropagation
    """
    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"damping": 0.5}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(AffinityPropagation, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnAffinityPropagation(**self._parameters)
