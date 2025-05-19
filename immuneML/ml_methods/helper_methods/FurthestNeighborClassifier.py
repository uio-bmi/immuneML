import numpy as np
from sklearn.neighbors import NearestNeighbors


class FurthestNeighborClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean', precomputed=False, **kwargs):
        """
        Parameters
        ----------
        n_neighbors : int
            Number of neighbors to consider.
        metric : str
            Distance metric or 'precomputed' for precomputed distance matrix.
        precomputed : bool
            If True, input X is a distance matrix.
        """
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.precomputed = precomputed
        self.distance_matrix = None

        self.nn = None
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        self.y_train = y
        self.classes_ = np.unique(y)

        if not self.precomputed:
            self.X_train = X
            self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
            self.nn.fit(X)
        else:
            # When precomputed, X is distance matrix
            self.X_train = None
            self.distance_matrix = X  # store full train-train distances
            if X.shape[0] != len(y):
                raise ValueError("Distance matrix size does not match number of training samples")

        return self

    def predict(self, X):
        # TODO: check if this makes any sense
        predictions = []

        if not self.precomputed:
            if self.nn is None:
                raise RuntimeError("You must fit the model before predicting")
            distances, indices = self.nn.kneighbors(X, return_distance=True)
        else:
            # X is test-train distance matrix, shape (n_test, n_train)
            distances = X
            # Find indices of k smallest distances for each test point
            indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]

            # Sort neighbors per row by distance
            sorted_idx = np.argsort(distances[np.arange(distances.shape[0])[:, None], indices], axis=1)
            indices = indices[np.arange(indices.shape[0])[:, None], sorted_idx]
            distances = distances[np.arange(distances.shape[0])[:, None], indices]

        for dist_row, idx_row in zip(distances, indices):
            labels = self.y_train[idx_row]
            class_to_max_dist = {}
            for c in self.classes_:
                class_dists = dist_row[labels == c]
                if len(class_dists) > 0:
                    class_to_max_dist[c] = np.max(class_dists)

            pred_class = min(class_to_max_dist, key=class_to_max_dist.get)
            predictions.append(pred_class)

        return np.array(predictions)
