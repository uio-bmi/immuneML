from immuneML.ml_methods.Clustering.Clustering import Clustering


class KMedoids(Clustering):
    """
    This is a wrapper of the KMedoids class from the scikit-learn-extra library for clustering. Please see the
    scikit-learn-extra documentation <https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html#sklearn_extra.cluster.KMedoids>
    of KMedoids for the parameters.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

    my_kmedoids: # user-defined method name
        KMedoids: # name of the Clustering method
            # sklearn-extra parameters (same names as in original sklearn-extra class)
            n_clusters: 2 # The number of clusters to form as well as the number of medoids to generate
            method: 'pam' # The method to be used: 'pam' or 'alternate'
            init: 'k-medoids++' # The initialization method to be used: 'random', 'heuristic' or 'k-medoids++'
    my_default_kmedoids: KMedoids
    """
    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {"n_clusters": 2, "method": "pam", "init": "k-medoids++"}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(KMedoids, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        from sklearn_extra.cluster import KMedoids as SklearnKMedoids
        return SklearnKMedoids(**self._parameters)
