import copy
import yaml
import numpy as np
from pathlib import Path
import random

from immuneML.environment.Label import Label
from immuneML.util.PathBuilder import PathBuilder
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.data_model.encoded_data.EncodedData import EncodedData


class SillyClassifier(MLMethod):
    """
    This SillyClassifier is a placeholder for a real ML method.
    It generates random predictions ignoring the input features.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_silly_method: SillyClassifier
    """

    def __init__(self):
        super().__init__()
        self.silly_model_fitted = False

    def _fit(self, encoded_data: EncodedData, cores_for_training: int = 2):
        # fitting does nothing, except set this parameter to true
        self.silly_model_fitted = True

    def _predict_proba(self, encoded_data: EncodedData):
        pred_probabilities = [random.random() for encoded_sequence in encoded_data.examples]
        pred_probabilities = np.array(pred_probabilities)

        # e.g., {gluten: {binder: [0.2, 0.4, 0.9], nonbinder: [0.8, 0.6, 0.1]}}
        return {self.label.name: {self.label.positive_class: pred_probabilities,
                                  self.label.get_binary_negative_class(): 1 - pred_probabilities}}

    def _predict(self, encoded_data: EncodedData):
        predictions_proba = self.predict_proba(encoded_data, self.label)
        proba_positive_class = predictions_proba[self.label.name][self.label.positive_class]

        predictions = []

        for proba in proba_positive_class:
            if proba > 0.5:
                predictions.append(self.label.positive_class)
            else:
                predictions.append(self.label.get_binary_negative_class())

        # e.g., {gluten: [nonbinder, nonbinder, binder]]}
        return {self.label.name: np.array(predictions)}

    def get_compatible_encoders(self):
        # Every encoder that is compatible with the ML method should be listed here.
        # The SillyClassifier can in principle be used with any encoder, few examples are listed
        from immuneML.encodings.distance_encoding.DistanceEncoder import DistanceEncoder
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder

        return [DistanceEncoder, KmerFrequencyEncoder, OneHotEncoder]

    def can_predict_proba(self) -> bool:
        return True

    def can_fit_with_example_weights(self) -> bool:
        return False

    def store(self, path: Path):
        # The most basic way of storing a model is to get the parameters in a yaml-friendly format (get_params)
        # and store this in a file.
        # Depending on the method, more files (e.g., internal pickle, pytorch or keras files) may need to be stored.
        # The 'store' method should be compatible with 'load'
        PathBuilder.build(path)
        class_parameters = self.get_params()
        params_path = path / "custom_params.yaml"

        with params_path.open('w') as file:
            yaml.dump(class_parameters, file)

    def get_params(self) -> dict:
        # Returns a yaml-friendly dictionary (only simple types, no objects) with all parameters of this ML method
        params = copy.deepcopy(vars(self))

        if self.label:
            # the 'Label' object must be converted to a yaml-friendly version
            params["label"] = self.label.get_desc_for_storage()

        return params

    def load(self, path: Path):
        # The 'load' method is called on a new (untrained) object of the specific MLMethod class.
        # This method is used to load parameters from a previously trained model from storage,
        # thus creating a copy of the original trained model

        # Load the dictionary of parameters from the YAML file
        params_path = path / "custom_params.yaml"

        with params_path.open("r") as file:
            custom_params = yaml.load(file, Loader=yaml.SafeLoader)

        # Loop through the dictionary and set each parameter
        for param, value in custom_params.items():
            if hasattr(self, param):
                if param == "label":
                    # Special case: if the parameter is 'label', convert to a Label object
                    setattr(self, "label", Label(**value))
                else:
                    # Other cases: directly set the parameter to the given value
                    setattr(self, param, value)
