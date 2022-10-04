import abc
import os
import warnings
import numpy as np
import tensorflow

from immuneML.ml_methods.util.Util import Util
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment import Label
from immuneML.ml_methods.MLMethod import MLMethod


class GenerativeModel(MLMethod):

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(GenerativeModel, self).__init__()
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
        self.label = None

    def fit(self, encoded_data: EncodedData, label: Label, cores_for_training: int = 2):

        self.label = label
        self.class_mapping = Util.make_class_mapping(encoded_data.labels[self.label.name])
        self.feature_names = encoded_data.feature_names

        mapped_y = Util.map_to_new_class_values(encoded_data.labels[self.label.name], self.class_mapping)

        self.model = self._fit(encoded_data.examples, mapped_y, cores_for_training)

    def predict(self, encoded_data: EncodedData, label: Label):
        #self.check_is_fitted(label.name)
        predictions = self.model.predict(encoded_data.examples)
        return {label.name: Util.map_to_old_class_values(np.array(predictions), self.class_mapping)}

    def predict_proba(self, encoded_data: EncodedData, label: Label):
        if self.can_predict_proba():
            predictions = {label.name: self.model.predict_proba(encoded_data.examples)}
            return predictions
        else:
            return None

    def _fit(self, X, y, cores_for_training: int = 1):
        if not self.show_warnings:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"

        print(X[:50])
        vocab_size = 21
        embedding_dim = 256
        batch_size = 64

        self.model = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape = [batch_size,None]),
            self._get_ml_model(cores_for_training, X),
            tensorflow.keras.layers.Dense(vocab_size)
        ])

        self.model.fit(X, y)

        if not self.show_warnings:
            del os.environ["PYTHONWARNINGS"]
            warnings.simplefilter("always")

        return self.model

    @abc.abstractmethod
    def _get_ml_model(self, cores_for_training: int = 2, X=None):

        pass

    @abc.abstractmethod
    def get_params(self):
        '''Returns the model parameters in a readable yaml-friendly way (consisting of lists, dictionaries and strings).'''
        pass

    @staticmethod
    def get_usage_documentation(model_name):
        return f"""
        
        TODO
        
        Following text does not relate to generative models

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

