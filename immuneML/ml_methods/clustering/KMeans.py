from sklearn.cluster import KMeans as SklearnKmeans

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod, get_data_for_clustering


class KMeans(ClusteringMethod):
    # TODO: add documentation!

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
