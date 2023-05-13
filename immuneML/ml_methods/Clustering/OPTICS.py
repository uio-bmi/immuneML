from sklearn.cluster import OPTICS as SklearnOPTICS

from immuneML.ml_methods.Clustering.Clustering import Clustering


class OPTICS(Clustering):
    """
    This is a wrapper of scikit-learn’s OPTICS class for clustering. Please see the
    scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html>
    of OPTICS for the parameters.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_optics: # user-defined method name
            OPTICS: # name of the Clustering method
                # sklearn parameters (same names as in original sklearn class)
                min_samples: 5 # The number of samples in a neighborhood for a point to be considered as a core point.
        my_default_optics: OPTICS
    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"min_samples": 5}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(OPTICS, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnOPTICS(**self._parameters)
