import abc
import json
import os
import pickle
import warnings
from collections import Iterable

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted

from source.environment.ParallelismManager import ParallelismManager
from source.ml_methods.MLMethod import MLMethod
from source.util.FilenameHandler import FilenameHandler
from source.util.PathBuilder import PathBuilder


class SklearnMethod(MLMethod):

    def __init__(self):
        self._models = {}
        self._parameter_grid = {}
        self._parameters = None

    def _fit_for_label(self, X: Iterable, y: np.ndarray, label: str, cores: int):
        self._models[label] = self._get_ml_model()
        self._models[label].fit(X, y)

    def fit(self, X: Iterable, y, label_names: list = None):
        cores = ParallelismManager.assign_cores_to_job()

        if label_names is not None:
            for index, label in enumerate(label_names):
                self._fit_for_label(X, y[label], label, cores)
        else:
            warnings.warn(
                "{}: label names not set, assuming only one and attempting to fit the model with label 'default'..."
                    .format(self.__class__.__name__),
                Warning)
            self._fit_for_label(X, y["default"], "default", cores)

        ParallelismManager.free_cores(cores=cores)

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

    def _fit_for_label_by_cv(self, X: Iterable, y: np.ndarray, label: str, cores: int, number_of_splits: int = 5):
        self._models[label] = RandomizedSearchCV(self._get_ml_model(cores_for_training=1),
                                                 param_distributions=self._parameter_grid,
                                                 cv=number_of_splits, n_jobs=cores,
                                                 scoring="balanced_accuracy", refit=True)
        self._models[label].fit(X, y)
        self._models[label] = self._models[label].best_estimator_  # do not leave RandomSearchCV object to be in models, but use the best estimator instead

    def fit_by_cross_validation(self, X, y, number_of_splits: int = 5, parameter_grid: dict = None,
                                label_names: list = None):
        if parameter_grid is not None:
            self._parameter_grid = parameter_grid

        n_jobs = ParallelismManager.assign_cores_to_job()

        for label in label_names:
            self._fit_for_label_by_cv(X, y[label], label, n_jobs, number_of_splits)

        ParallelismManager.free_cores(cores=n_jobs)

    def store(self, path):
        PathBuilder.build(path)
        name = FilenameHandler.get_filename(self.__class__.__name__, "pickle")
        params_name = FilenameHandler.get_filename(self.__class__.__name__, "json")
        with open(path + name, "wb") as file:
            pickle.dump(self._models, file)
        with open(path + params_name, "w") as file:
            desc = {label: self._models[label].estimator.get_params()
                    if isinstance(self._models[label], RandomizedSearchCV)
                    else self._models[label].get_params() for label in self._models.keys()}
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
