from sklearn.cluster import KMeans as SklearnKmeans

from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod


class KMeans(ClusteringMethod):

    def __init__(self, name=None, **kwargs):
        super().__init__(name)
        self.model = SklearnKmeans(**kwargs)
