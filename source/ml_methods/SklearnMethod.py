import abc
import hashlib
import json
import os
import pickle
import warnings
from collections import Iterable

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted

from source.caching.CacheHandler import CacheHandler
from source.ml_methods.MLMethod import MLMethod
from source.util.FilenameHandler import FilenameHandler
from source.util.PathBuilder import PathBuilder


class SklearnMethod(MLMethod):

    FIT_CV = "fit_CV"
    FIT = "fit"

    def __init__(self):
        self._models = {}
        self._parameter_grid = {}
        self._parameters = None

    def _fit_for_label(self, X: Iterable, y: np.ndarray, label: str, cores_for_training: int):
        self._models[label] = self._get_ml_model(cores_for_training)
        self._models[label].fit(X, y)

    def _prepare_caching_params(self, X: Iterable, y, type: str, label_names: list = None, number_of_splits: int = -1):
        return (("X", hashlib.sha256(str(X).encode("utf-8")).hexdigest()),
                ("y", hashlib.sha256(str(y).encode("utf-8")).hexdigest()),
                ("label_names", str(label_names)),
                ("type", type),
                ("number_of_splits", str(number_of_splits)),
                ("parameters", str(self._parameters)),
                ("parameter_grid", str(self._parameter_grid)),)

    def fit(self, X: Iterable, y, label_names: list = None, cores_for_training: int = 1):

        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(X, y, self.FIT, label_names))
        CacheHandler.memo(cache_key, lambda: self._fit(X, y, label_names, cores_for_training))

    def _fit(self, X: Iterable, y, label_names: list = None, cores_for_training: int = 1):

        if label_names is not None:
            for index, label in enumerate(label_names):
                self._fit_for_label(X, y[label], label, cores_for_training)
        else:
            warnings.warn(
                "{}: label names not set, assuming only one and attempting to fit the model with label 'default'..."
                    .format(self.__class__.__name__),
                Warning)
            self._fit_for_label(X, y["default"], "default", cores_for_training)

    def _can_predict_proba(self) -> bool:
        return False

    def check_is_fitted(self, labels):
        return all([check_is_fitted(self._models[label], ["estimators_", "coef_", "estimator"], all_or_any=any) for label in labels])

    def predict(self, X: Iterable, label_names: list = None):
        labels = label_names if label_names is not None else self._models.keys()
        self.check_is_fitted(labels)
        return {label: self._models[label].predict(X) for label in labels}

    def predict_proba(self, X: Iterable, labels: list):
        if self._can_predict_proba():
            predictions = {label: self._models[label].predict_proba(X) for label in labels}
            return predictions
        else:
            return None

    def _fit_for_label_by_cv(self, X: Iterable, y: np.ndarray, label: str, cores_for_training: int, number_of_splits: int = 5):
        self._models[label] = RandomizedSearchCV(self._get_ml_model(cores_for_training=cores_for_training),
                                                 param_distributions=self._parameter_grid,
                                                 cv=number_of_splits, n_jobs=cores_for_training,
                                                 scoring="balanced_accuracy", refit=True)
        self._models[label].fit(X, y)
        self._models[label] = self._models[label].best_estimator_  # do not leave RandomSearchCV object to be in models, but use the best estimator instead

    def fit_by_cross_validation(self,  X, y, number_of_splits: int = 5, parameter_grid: dict = None,
                                label_names: list = None, cores_for_training: int = 1):

        if parameter_grid is not None:
            self._parameter_grid = parameter_grid

        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(X, y, self.FIT_CV, label_names,
                                                                                 number_of_splits))
        CacheHandler.memo(cache_key, lambda: self._fit(X, y, label_names, cores_for_training))

    def _fit_by_cross_validation(self, X, y, number_of_splits: int = 5, label_names: list = None,
                                 cores_for_training: int = 1):

        for label in label_names:
            self._fit_for_label_by_cv(X, y[label], label, cores_for_training, number_of_splits)

    def store(self, path, feature_names=None):
        PathBuilder.build(path)
        name = FilenameHandler.get_filename(self.__class__.__name__, "pickle")
        params_name = FilenameHandler.get_filename(self.__class__.__name__, "json")
        with open(path + name, "wb") as file:
            pickle.dump(self._models, file)
        with open(path + params_name, "w") as file:
            desc = {}
            for label in self._models.keys():
                desc[label] = {
                    **(self.get_params(label)),
                    "feature_names": feature_names,
                    "classes": self._models[label].classes_.tolist()
                }
            json.dump(desc, file, indent=2)

    def load(self, path):
        name = FilenameHandler.get_filename(self.__class__.__name__, "pickle")
        if os.path.isfile(path + name):
            with open(path + name, "rb") as file:
                self._models = pickle.load(file)
        else:
            raise FileNotFoundError(self.__class__.__name__ + " model could not be loaded from " + str(
                path + name) + ". Check if the path to the " + name + " file is properly set.")

    def get_model(self, label_names: list = None):
        if label_names is None:
            return self._models
        else:
            return {key: self._models[key] for key in self._models.keys() if key in label_names}

    def get_classes_for_label(self, label):
        return self._models[label].classes_

    def check_if_exists(self, path):
        return os.path.isfile(path + FilenameHandler.get_filename(self.__class__.__name__, "pickle"))

    @abc.abstractmethod
    def _get_ml_model(self, cores_for_training: int = 2):
        pass
