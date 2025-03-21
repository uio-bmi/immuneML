from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod, get_data_for_clustering
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerativeClustering


class AgglomerativeClustering(ClusteringMethod):
    """
    Agglomerative clustering method which wraps scikit-learn's clustering of the same name.
    Input arguments for the method are the same as supported by scikit-learn (see `AgglomerativeClustering scikit-learn documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_ for details).

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_agglomerative_clustering:
                    AgglomerativeClustering:
                        # arguments as defined by scikit-learn
                        n_clusters: 3
                        linkage: 'ward'
        """

    def __init__(self, name=None, **kwargs):
        super().__init__(name)
        self.model = SklearnAgglomerativeClustering(**kwargs)

    def fit(self, dataset: Dataset):
        data = get_data_for_clustering(dataset)
        self.model.fit(data)

    def fit_predict(self, dataset: Dataset):
        data = get_data_for_clustering(dataset)
        return self.model.fit_predict(data)

    def predict(self, dataset: Dataset):
        raise RuntimeError("Agglomerative Clustering does not support predict method")

    def transform(self, dataset: Dataset):
        raise RuntimeError("Agglomerative Clustering does not support transform method")
