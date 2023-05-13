from sklearn.cluster import AgglomerativeClustering as AgglomerativeClustering
from immuneML.ml_methods.Clustering.Clustering import Clustering


class Agglomerative(Clustering):
    """
    This is a wrapper of scikit-learn’s AgglomerativeClustering class for clustering. Please see the
    scikit-learn documentation https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#
    of AgglomerativeClustering for the parameters.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

    my_agglomerative: # user-defined method name
        Agglomerative: # name of the Clustering method
            # sklearn parameters (same names as in original sklearn class)
            n_clusters: 5 # The number of clusters to find
    my_default_agglomerative: Agglomerative

    """
    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"n_clusters": 2}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(Agglomerative, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return AgglomerativeClustering(**self._parameters)

    def get_compatible_encoders(self):
        encodings = super().get_compatible_encoders()
        from immuneML.encodings.distance_encoding.TCRdistEncoder import TCRdistEncoder

        encodings.append(TCRdistEncoder)
        return encodings
