from sklearn.cluster import KMeans as SklearnKmeans

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod, get_data_for_clustering


class KMeans(ClusteringMethod):
    '''
    k-means clustering method which wraps scikit-learn's KMeans. Input arguments for the method are the
    same as supported by scikit-learn (see `KMeans scikit-learn documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ for details).

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_kmeans:
                    KMeans:
                        # arguments as defined by scikit-learn
                        n_clusters: 2
    '''

    def __init__(self, name=None, **kwargs):
        super().__init__(name)
        self.model = SklearnKmeans(**kwargs)

    def fit(self, dataset: Dataset):
        data = get_data_for_clustering(dataset)
        self.model.fit(data)

    def predict(self, dataset: Dataset):
        data = get_data_for_clustering(dataset)
        return self.model.predict(data)

    def transform(self, dataset: Dataset):
        data = get_data_for_clustering(dataset)
        return self.model.transform(data)
