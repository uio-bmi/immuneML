import abc
import os
import warnings
from pathlib import Path

import dill
import numpy as np
import pkg_resources
import yaml
from sklearn.metrics import SCORERS
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.util.Util import Util
from immuneML.util.FilenameHandler import FilenameHandler
from immuneML.util.PathBuilder import PathBuilder


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

        parameters: a dictionary of parameters that will be directly passed to scikit-learn's class upon calling __init__()
            method; for detailed list see scikit-learn's documentation of the specific class inheriting SklearnMethod

        parameter_grid: a dictionary of parameters which all have to be valid arguments for scikit-learn's corresponding class' __init__() method
            (same as parameters), but unlike parameters argument can contain list of values instead of one value; if this is specified and
            "model_selection_cv" is True (in the specification) or just if fit_by_cross_validation() is called, a grid search will be performed over
            these parameters and the optimal model will be kept

    YAML specification:

        ml_methods:
            log_reg:
                LogisticRegression: # name of the class inheriting SklearnMethod
                    # sklearn parameters (same names as in original sklearn class)
                    max_iter: 1000 # specific parameter value
                    penalty: l1
                    # Additional parameter that determines whether to print convergence warnings
                    show_warnings: True
                # if any of the parameters under LogisticRegression is a list and model_selection_cv is True,
                # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
                # and the optimal model will be selected
                model_selection_cv: True
                model_selection_n_folds: 5
            svm_with_cv:
                SVM: # name of another class inheriting SklearnMethod
                    # sklearn parameters (same names as in original sklearn class)
                    alpha: 10
                    # Additional parameter that determines whether to print convergence warnings
                    show_warnings: True
                # no grid search will be done
                model_selection_cv: False

    """

    FIT_CV = "fit_CV"
    FIT = "fit"

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(SklearnMethod, self).__init__()
        self.model = None

        if parameter_grid is not None and "show_warnings" in parameter_grid:
            self.show_warnings = parameter_grid.pop("show_warnings")[0]
        elif parameters is not None and "show_warnings" in parameters:
            self.show_warnings = parameters.pop("show_warnings")
        else:
            self.show_warnings = True

        self._parameter_grid = parameter_grid
        self._parameters = parameters
        self.feature_names = None
        self.class_mapping = None
        self.label_name = None

    def fit(self, encoded_data: EncodedData, label_name: str, cores_for_training: int = 2):

        self.class_mapping = Util.make_class_mapping(encoded_data.labels[label_name])
        self.feature_names = encoded_data.feature_names
        self.label_name = label_name

        mapped_y = Util.map_to_new_class_values(encoded_data.labels[label_name], self.class_mapping)

        self.model = self._fit(encoded_data.examples, mapped_y, cores_for_training)

    def predict(self, encoded_data: EncodedData, label_name: str):
        self.check_is_fitted(label_name)
        predictions = self.model.predict(encoded_data.examples)
        return {label_name: Util.map_to_old_class_values(np.array(predictions), self.class_mapping)}

    def predict_proba(self, encoded_data: EncodedData, label_name: str):
        if self.can_predict_proba():
            predictions = {label_name: self.model.predict_proba(encoded_data.examples)}
            return predictions
        else:
            return None

    def _fit(self, X, y, cores_for_training: int = 1):
        if not self.show_warnings:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"

        self.model = self._get_ml_model(cores_for_training, X)
        self.model.fit(X, y)

        if not self.show_warnings:
            del os.environ["PYTHONWARNINGS"]
            warnings.simplefilter("always")

        return self.model

    def can_predict_proba(self) -> bool:
        return False

    def check_is_fitted(self, label_name):
        if self.label_name == label_name or label_name is None:
            return check_is_fitted(self.model, ["estimators_", "coef_", "estimator", "_fit_X", "dual_coef_"], all_or_any=any)

    def fit_by_cross_validation(self, encoded_data: EncodedData, number_of_splits: int = 5, label_name: str = None, cores_for_training: int = -1,
                                optimization_metric='balanced_accuracy'):

        self.class_mapping = Util.make_class_mapping(encoded_data.labels[label_name])
        self.feature_names = encoded_data.feature_names
        self.label_name = label_name
        mapped_y = Util.map_to_new_class_values(encoded_data.labels[label_name], self.class_mapping)

        self.model = self._fit_by_cross_validation(encoded_data.examples, mapped_y, number_of_splits, label_name, cores_for_training,
                                                  optimization_metric)

    def _fit_by_cross_validation(self, X, y, number_of_splits: int = 5, label_name: str = None, cores_for_training: int = 1,
                                 optimization_metric: str = "balanced_accuracy"):

        model = self._get_ml_model()
        scoring = optimization_metric

        if optimization_metric not in SCORERS.keys():
            scoring = "balanced_accuracy"
            warnings.warn(
                f"{self.__class__.__name__}: specified optimization metric ({optimization_metric}) is not defined as a sklearn scoring function, using {scoring} instead... ")

        if not self.show_warnings:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"

        self.model = RandomizedSearchCV(model, param_distributions=self._parameter_grid, cv=number_of_splits, n_jobs=cores_for_training,
                                        scoring=scoring, refit=True)
        self.model.fit(X, y)

        if not self.show_warnings:
            del os.environ["PYTHONWARNINGS"]
            warnings.simplefilter("always")

        self.model = self.model.best_estimator_  # do not leave RandomSearchCV object to be in models, use the best estimator instead

        return self.model

    def store(self, path: Path, feature_names=None, details_path: Path = None):
        PathBuilder.build(path)
        file_path = path / f"{self._get_model_filename()}.pickle"
        with file_path.open("wb") as file:
            dill.dump(self.model, file)

        if details_path is None:
            params_path = path / f"{self._get_model_filename()}.yaml"
        else:
            params_path = details_path

        with params_path.open("w") as file:
            desc = {
                **(self.get_params()),
                "feature_names": feature_names,
                "classes": self.model.classes_.tolist(),
                "class_mapping": self.class_mapping,
                "label": self.label_name
            }
            yaml.dump(desc, file)

    def _get_model_filename(self):
        return FilenameHandler.get_filename(self.__class__.__name__, "")

    def load(self, path: Path, details_path: Path = None):
        name = f"{self._get_model_filename()}.pickle"
        file_path = path / name
        if file_path.is_file():
            with file_path.open("rb") as file:
                self.model = dill.load(file)
        else:
            raise FileNotFoundError(f"{self.__class__.__name__} model could not be loaded from {file_path}"
                                    f". Check if the path to the {name} file is properly set.")

        if details_path is None:
            params_path = path / f"{self._get_model_filename()}.yaml"
        else:
            params_path = details_path

        if params_path.is_file():
            with params_path.open("r") as file:
                desc = yaml.safe_load(file)
                for param in ["feature_names", "classes", "class_mapping", "label"]:
                    if param in desc:
                        setattr(self, param, desc[param])

    def get_model(self):
        return self.model

    def get_classes(self):
        return list(self.class_mapping.values())

    def check_if_exists(self, path: Path):
        file_path = path / f"{self._get_model_filename()}.pickle"
        return file_path.is_file()

    @abc.abstractmethod
    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        pass

    @abc.abstractmethod
    def get_params(self):
        pass

    def get_label(self):
        return self.label_name

    def get_package_info(self) -> str:
        return 'scikit-learn ' + pkg_resources.get_distribution('scikit-learn').version

    def get_feature_names(self) -> list:
        return self.feature_names

    def get_class_mapping(self) -> dict:
        """Returns a dictionary containing the mapping between label values and values internally used in the classifier"""
        return self.class_mapping

    def get_compatible_encoders(self):
        from immuneML.encodings.evenness_profile.EvennessProfileEncoder import EvennessProfileEncoder
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
        from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
        return [KmerFrequencyEncoder, OneHotEncoder, Word2VecEncoder, EvennessProfileEncoder]

    @staticmethod
    def get_usage_documentation(model_name):
        return f"""
    
    Scikit-learn models can be trained in two modes: 
    
    1. Creating a model using a given set of hyperparameters, and relying on the selection and assessment loop in the
    TrainMLModel instruction to select the optimal model. 
    
    2. Passing a range of different hyperparameters to {model_name}, and using a third layer of nested cross-validation 
    to find the optimal hyperparameters through grid search. In this case, only the {model_name} model with the optimal 
    hyperparameter settings is further used in the inner selection loop of the TrainMLModel instruction. 
    
    By default, mode 1 is used. In order to use mode 2, model_selection_cv and model_selection_n_folds must be set. 
    
    
    Arguments:

        {model_name} (dict): Under this key, hyperparameters can be specified that will be passed to the scikit-learn class.
        Any scikit-learn hyperparameters can be specified here. In mode 1, a single value must be specified for each of the scikit-learn
        hyperparameters. In mode 2, it is possible to specify a range of different hyperparameters values in a list. It is also allowed
        to mix lists and single values in mode 2, in which case the grid search will only be done for the lists, while the
        single-value hyperparameters will be fixed. 
        In addition to the scikit-learn hyperparameters, parameter show_warnings (True/False) can be specified here. This determines
        whether scikit-learn warnings, such as convergence warnings, should be printed. By default show_warnings is True.
        
        model_selection_cv (bool): If any of the hyperparameters under {model_name} is a list and model_selection_cv is True, 
        a grid search will be done over the given hyperparameters, using the number of folds specified in model_selection_n_folds.
        By default, model_selection_cv is False. 
        
        model_selection_n_folds (int): The number of folds that should be used for the cross validation grid search if model_selection_cv is True.
        
        """
