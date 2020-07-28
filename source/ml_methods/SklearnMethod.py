import abc
import hashlib
import os
import warnings

import dill
import numpy as np
import yaml
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted

from source.caching.CacheHandler import CacheHandler
from source.data_model.encoded_data.EncodedData import EncodedData
from source.ml_methods.MLMethod import MLMethod
from source.util.FilenameHandler import FilenameHandler
from source.util.PathBuilder import PathBuilder


class SklearnMethod(MLMethod):
    """
    Base class for ML methods imported from scikit-learn. The classes inheriting SklearnMethod acting as wrappers around imported
    ML methods from scikit-learn have to implement:
        - the __init__() method,
        - get_params(label) and
        - _get_ml_model()
    Other methods can also be overwritten if needed.
    The arguments and specification described bellow applied for all classes inheriting SklearnMethod.

    Arguments:
        parameters: a dictionary of parameters that will be directly passed to the scikit-learn's LogisticRegression class
            upon calling __init__() method; for detailed list see scikit-learn's documentation of the specific class
            inheriting SklearnMethod
        parameter_grid: a dictionary of parameters which all have to be valid arguments for scikit-learn's corresponding class'
            __init__() method (same as parameters), but unlike parameters argument can contain list of values instead of one value;
            if this is specified and "model_selection_cv" is True (in the specification) or just if fit_by_cross_validation() is called,
            a grid search will be performed over these parameters and the optimal model will be kept

    Specification:
        ml_methods:
            log_reg:
                SimpleLogisticRegression: # name of the class inheriting SklearnMethod
                    max_iter: 1000 # specific parameter value
                    penalty: l1
                # if any of the parameters under SimpleLogisticRegression is a list and model_selection_cv is True,
                # SimpleLogisticRegression will do grid search over the given parameters and return optimal model
                model_selection_cv: False
                model_selection_n_folds: -1
            svm_with_cv:
                SVM: # name of another class inheriting SklearnMethod
                    alpha: [10, 100] # search will be performed over these parameters
                # if any of the parameters under SVM is a list and model_selection_cv is True,
                # SVM will do grid search over the given parameters and return optimal model
                model_selection_cv: True
                model_selection_n_folds: 5
    """

    FIT_CV = "fit_CV"
    FIT = "fit"

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(SklearnMethod, self).__init__()
        self.models = {}
        self._parameter_grid = {}
        self._parameters = None

    def _fit_for_label(self, X, y: np.ndarray, label: str, cores_for_training: int):
        self.models[label] = self._get_ml_model(cores_for_training, X)
        self.models[label].fit(X, y)

    def _prepare_caching_params(self, encoded_data: EncodedData, y, type: str, label_names: list = None, number_of_splits: int = -1):
        return (("encoded_data", hashlib.sha256(str(encoded_data.examples).encode("utf-8")).hexdigest()),
                ("y", hashlib.sha256(str(y).encode("utf-8")).hexdigest()),
                ("label_names", str(label_names)),
                ("type", type),
                ("number_of_splits", str(number_of_splits)),
                ("parameters", str(self._parameters)),
                ("parameter_grid", str(self._parameter_grid)),)

    def fit(self, encoded_data: EncodedData, y, label_names: list = None, cores_for_training: int = 1):
        self.models = CacheHandler.memo_by_params(self._prepare_caching_params(encoded_data, y, self.FIT, label_names),
                                                  lambda: self._fit(encoded_data.examples, y, label_names, cores_for_training))

    def _fit(self, X, y, label_names: list = None, cores_for_training: int = 1):

        if label_names is not None:
            for index, label in enumerate(label_names):
                self._fit_for_label(X, y[label], label, cores_for_training)
        else:
            warnings.warn(
                "{}: label names not set, assuming only one and attempting to fit the model with label 'default'..."
                    .format(self.__class__.__name__),
                Warning)
            self._fit_for_label(X, y["default"], "default", cores_for_training)

        return self.models

    def _can_predict_proba(self) -> bool:
        return False

    def check_is_fitted(self, labels):
        return all([check_is_fitted(self.models[label], ["estimators_", "coef_", "estimator", "_fit_X"], all_or_any=any) for label in labels])

    def predict(self, encoded_data: EncodedData, label_names: list = None):
        labels = label_names if label_names is not None else self.models.keys()
        self.check_is_fitted(labels)
        return {label: self.models[label].predict(encoded_data.examples) for label in labels}

    def predict_proba(self, encoded_data: EncodedData, labels: list):
        if self._can_predict_proba():
            predictions = {label: self.models[label].predict_proba(encoded_data.examples) for label in labels}
            return predictions
        else:
            return None

    def _fit_for_label_by_cv(self, X, y: np.ndarray, label: str, cores_for_training: int, number_of_splits: int = 5):
        self.models[label] = RandomizedSearchCV(self._get_ml_model(cores_for_training=cores_for_training),
                                                param_distributions=self._parameter_grid,
                                                cv=number_of_splits, n_jobs=cores_for_training,
                                                scoring="balanced_accuracy", refit=True)
        self.models[label].fit(X, y)
        self.models[label] = self.models[
            label].best_estimator_  # do not leave RandomSearchCV object to be in models, but use the best estimator instead

    def fit_by_cross_validation(self, encoded_data: EncodedData, y, number_of_splits: int = 5, parameter_grid: dict = None,
                                label_names: list = None, cores_for_training: int = 1):

        if parameter_grid is not None:
            self._parameter_grid = parameter_grid

        self.models = CacheHandler.memo_by_params(self._prepare_caching_params(encoded_data, y, self.FIT_CV, label_names,
                                                                               number_of_splits),
                                                  lambda: self._fit_by_cross_validation(encoded_data.examples, y, number_of_splits,
                                                                                        label_names, cores_for_training))

    def _fit_by_cross_validation(self, X, y, number_of_splits: int = 5, label_names: list = None,
                                 cores_for_training: int = 1):

        for label in label_names:
            self._fit_for_label_by_cv(X, y[label], label, cores_for_training, number_of_splits)

        return self.models

    def store(self, path, feature_names=None, details_path=None):
        PathBuilder.build(path)
        name = FilenameHandler.get_filename(self.__class__.__name__, "pickle")
        with open(path + name, "wb") as file:
            dill.dump(self.models, file)

        if details_path is None:
            params_path = path + FilenameHandler.get_filename(self.__class__.__name__, "yaml")
        else:
            params_path = details_path

        with open(params_path, "w") as file:
            desc = {}
            for label in self.models.keys():
                desc[label] = {
                    **(self.get_params(label)),
                    "feature_names": feature_names,
                    "classes": self.models[label].classes_.tolist()
                }
            yaml.dump(desc, file)

    def load(self, path):
        name = FilenameHandler.get_filename(self.__class__.__name__, "pickle")
        if os.path.isfile(path + name):
            with open(path + name, "rb") as file:
                self.models = dill.load(file)
        else:
            raise FileNotFoundError(self.__class__.__name__ + " model could not be loaded from " + str(
                path + name) + ". Check if the path to the " + name + " file is properly set.")

    def get_model(self, label_names: list = None):
        if label_names is None:
            return self.models
        else:
            return {key: self.models[key] for key in self.models.keys() if key in label_names}

    def get_classes_for_label(self, label):
        return self.models[label].classes_

    def check_if_exists(self, path):
        return os.path.isfile(path + FilenameHandler.get_filename(self.__class__.__name__, "pickle"))

    @abc.abstractmethod
    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        pass

    @abc.abstractmethod
    def get_params(self, label):
        pass

    def get_labels(self):
        return list(self.models.keys())
