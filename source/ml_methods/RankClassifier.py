import json
import os
import pickle

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from source.ml_methods.MLMethod import MLMethod
from source.util.FilenameHandler import FilenameHandler
from source.util.PathBuilder import PathBuilder


class RankClassifier(MLMethod):
    """
    Sorts all examples using the feature (all examples are represented by one feature) and tries to fit a threshold
    to maximize balanced accuracy for the binary classification.

    Supports multi-label, but not multi-class classification. In case of multi-label, it fits one threshold per label.

    Labels can have any value and will be transformed to standard reperesentation using sklearn's LabelEncoder. However,
    outputs of the model (e.g. predictions) will consist of original labels.s
    """
    def __init__(self, parameters=None):
        self.models = {}
        self._label_encoders = {}
        # a name of any valid binary operator function in numpy:
        self._parameters = {**{"comparison": "greater_equal"}, **(parameters if parameters is not None else {})}

    def _rank_examples(self, X, y):
        assert X.shape[1] == 1, "RankClassifier: examples should contain only one feature - score."

        sorted_indices = np.argsort(-X[:, 0])
        sorted_X = X[sorted_indices]
        sorted_y = {label: y[label][sorted_indices] for label in y.keys()}
        return sorted_X, sorted_y

    def _check_labels(self, label_names: list, y):
        for label in label_names:
            _, class_count = np.unique(y[label], return_counts=True)
            assert len(class_count) == 2, "RankClassifier: only multi-label classification is supported, but not multi-class. Issue occurred with label {}.".format(label)

    def _transform_labels(self, y, label_names: list = None) -> dict:
        encoded_y = {}
        for label in label_names if label_names is not None else y.keys():
            self._label_encoders[label] = LabelEncoder()
            encoded_y[label] = self._label_encoders[label].fit_transform(y[label])
        return encoded_y

    def fit(self, X, y: dict, label_names: list = None, cores_for_training: int = 2):
        self._check_labels(label_names, y)
        transformed_y = self._transform_labels(y, label_names)
        self._fit(X, transformed_y, label_names)

    def _fit(self, X, y, label_names: list = None):
        sorted_X, sorted_y = self._rank_examples(X, y)
        for label in label_names:
            self._fit_for_label(label, sorted_X, sorted_y)

    def _fit_for_label(self, label: str, sorted_X, sorted_y: dict):
        max_balanced_accuracy = -1
        self.models[label] = {"threshold": float(np.min(sorted_X))}

        for index in range(sorted_X.shape[0]):

            test_threshold = sorted_X[index]
            balanced_accuracy = balanced_accuracy_score(getattr(np, self._parameters["comparison"])
                                                        (sorted_X, test_threshold),
                                                        sorted_y[label])

            if balanced_accuracy > max_balanced_accuracy:
                max_balanced_accuracy = balanced_accuracy
                self.models[label]["threshold"] = float((sorted_X[index] + sorted_X[index+1]) / 2) if index < sorted_X.shape[0]-1 else float(sorted_X[index])

    def predict(self, X, label_names: list = None):
        predictions = {}
        for label in label_names if label_names is not None else ["default"]:
            predictions[label] = getattr(np, self._parameters["comparison"])(X, self.models[label]["threshold"])
            if label in self._label_encoders and not set.issubset(set(np.unique(predictions[label])), set(self._label_encoders[label].classes_)):
                predictions[label] = self._label_encoders[label].inverse_transform(predictions[label][:,0])
        return predictions

    def fit_by_cross_validation(self, X, y: dict, number_of_splits: int = 5, parameter_grid: dict = None,
                                label_names: list = None):
        return self.fit(X, y, label_names)

    def store(self, path, features_names):
        PathBuilder.build(path)
        name = FilenameHandler.get_filename(self.__class__.__name__, "pickle")
        params_name = FilenameHandler.get_filename(self.__class__.__name__, "json")
        with open(path + name, "wb") as file:
            pickle.dump(self.models, file)
        with open(path + params_name, "w") as file:
            desc = self.models
            json.dump(desc, file, indent=2)

    def load(self, path):
        name = FilenameHandler.get_filename(self.__class__.__name__, "pickle")
        if os.path.isfile(path + name):
            with open(path + name, "rb") as file:
                self.models = pickle.load(file)
        else:
            raise FileNotFoundError(self.__class__.__name__ + " model could not be loaded from " + str(
                path + name) + ". Check if the path to the " + name + " file is properly set.")

    def get_model(self, label_names: list = None):
        if label_names is None:
            return self.models
        else:
            return {label: self.models[label] for label in label_names}

    def check_if_exists(self, path):
        return os.path.isfile(path + FilenameHandler.get_filename(self.__class__.__name__, "pickle"))

    def get_classes_for_label(self, label):
        return self._label_encoders[label].classes_

    def get_params(self, label):
        return self.models[label]

    def predict_proba(self, X, labels):
        return None
