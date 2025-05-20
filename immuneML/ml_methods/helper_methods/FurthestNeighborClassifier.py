import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric


class FurthestNeighborClassifier:
    """
    Furthest Neighbor Classifier for clustering tasks. It predicts the label (cluster) of an example as the label
    corresponding to the minimal maximum distance across all labels in training data. The metric used for distance
    computation can be any metric supported by sklearn.metrics.DistanceMetric, or precomputed (e.g., when using
    TCRdistEncoder).

    """
    def __init__(self, metric: str = 'precomputed', **kwargs):
        super().__init__(**kwargs)
        self.metric = metric

        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        self.y_train = y
        self.classes_ = np.unique(y)
        self.X_train = X

        if self.metric == 'precomputed':
            assert X.shape[0] == X.shape[1], \
                (f"{FurthestNeighborClassifier.__name__}: distance matrix must be square for precomputed metric, "
                 f"got: {X.shape}.")

        return self

    def predict(self, X):
        if self.metric != 'precomputed':
            try:
                distance_metric = DistanceMetric.get_metric(self.metric)
            except Exception as e:
                raise ValueError(f"{FurthestNeighborClassifier.__name__}: Metric '{self.metric}' couldn't be "
                                 f"computed. Full error: {e}")
            distances = distance_metric.pairwise(X, self.X_train)
        else:
            distances = X

        max_dist_per_cluster = pd.DataFrame(
            {cluster: distances[:, self.y_train == cluster].max(axis=1) for cluster in self.classes_})
        predictions = max_dist_per_cluster.idxmin(axis=1).values

        return predictions
