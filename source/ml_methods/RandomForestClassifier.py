import json
import os
import pickle
import warnings
from collections import Iterable

from scipy import sparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import RandomizedSearchCV

from source.environment.ParallelismManager import ParallelismManager
from source.ml_methods.MLMethod import MLMethod
from source.util.PathBuilder import PathBuilder


class RandomForestClassifier(MLMethod):

    def __init__(self, parameter_grid: dict = None):
        if parameter_grid is not None:
            self.__parameter_grid = parameter_grid
        else:
            self.__parameter_grid = {"n_estimators": [10, 50, 100]}

        self.__models = {}

    def get_model(self, label_names: list = None):
        if label_names is None:
            return self.__models
        else:
            return {key: self.__models[key] for key in self.__models.keys() if key in label_names}

    def fit(self, X: Iterable, y: np.ndarray, label_names: list = None):
        cores = ParallelismManager.assign_cores_to_job()

        if label_names is not None and len(label_names) == 1:
            self.__fit_for_label(X, y, label_names[0], cores)
        elif label_names is not None and len(label_names) > 1:
            for index, label in enumerate(label_names):
                self.__fit_for_label(X, y[index], label, cores)
        else:
            warnings.warn(
                "RandomForestClassifier: label names not set, assuming only one and attempting to fit the model with label 'default'...",
                Warning)
            self.__fit_for_label(X, y, "default", cores)

        ParallelismManager.free_cores(cores=cores)

    def __fit_for_label(self, X: Iterable, y: np.ndarray, label: str, cores: int):
        self.__models[label] = RFC(n_jobs=cores)
        self.__models[label].fit(X, y)

    def predict(self, X: Iterable, label_names: list = None):
        labels = label_names if label_names is not None else self.__models.keys()
        assert all([isinstance(self.__models[label],
                               RFC) or isinstance(self.__models[label], RandomizedSearchCV) for label in
                    labels]), "RandomForestClassifier: The model has not yet been trained. First call fit() or fit_by_cross_validation() and then call predict(). Another option is to load an existing model from the corresponding pickle file."
        return {label: self.__models[label].predict(X) for label in labels}

    def fit_by_cross_validation(self, X: Iterable, y: np.ndarray, number_of_splits: int = 5, parameter_grid: dict = None, label_names: list = None):
        if parameter_grid is not None:
            self.__parameter_grid = parameter_grid

        n_jobs = ParallelismManager.assign_cores_to_job()

        for index, label in enumerate(label_names):
            self.__fit_for_label_by_cv(X, y[index], label, n_jobs, number_of_splits)

        ParallelismManager.free_cores(cores=n_jobs)

    def __fit_for_label_by_cv(self, X: Iterable, y: np.ndarray, label: str, cores: int, number_of_splits: int = 5):
        self.__models[label] = RandomizedSearchCV(RFC(),
                                                  param_distributions=self.__parameter_grid,
                                                  cv=number_of_splits, n_jobs=cores,
                                                  scoring="balanced_accuracy", refit=True)
        self.__models[label].fit(X, y)

    def store(self, path):
        PathBuilder.build(path)
        with open(path + "random_forest_classifier.pkl", "wb") as file:
            pickle.dump(self.__models, file)
        with open(path + "rfc_optimal_params.json", "w") as file:
            desc = {label: self.__models[label].estimator.get_params()
                    if isinstance(self.__models[label], RandomizedSearchCV)
                    else self.__models[label].get_params() for label in self.__models.keys()}
            json.dump(desc, file, indent=2)

    def load(self, path):
        if os.path.isfile(path + "random_forest_classifier.pkl"):
            with open(path + "random_forest_classifier.pkl", "rb") as file:
                self.__models = pickle.load(file)
        else:
            raise FileNotFoundError("Random forest classifier model could not be loaded from " + str(path) + "random_forest_classifier.pkl. Check if the path to the random_forest_classifier.pkl file is properly set.")

    def check_if_exists(self, path):
        return os.path.isfile(path + "random_forest_classifier.pkl")
