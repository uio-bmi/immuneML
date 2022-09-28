import copy
import logging
import math
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.pytorch_implementations.PyTorchReceptorCNN import PyTorchReceptorCNN as RCNN
from immuneML.ml_methods.util.Util import Util
from immuneML.util.PathBuilder import PathBuilder


class MotifClassifier(MLMethod):
    """

    Arguments:

        training_percentage (float): what percentage of data to use for training (the rest will be used for validation); values between 0 and 1



    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_motif_classifier:
            MotifClassifier:
                ...

    """

    def __init__(self, training_percentage: float = None, max_motifs: int = None,
                 patience: int = None, min_delta: float = None, optimization_metric: str = None,
                 result_path: Path = None):
        super().__init__()
        self.training_percentage = training_percentage
        self.max_motifs = max_motifs
        self.patience = patience
        self.min_delta = min_delta
        self.optimization_metric = optimization_metric

        self.class_mapping = None
        self.result_path = result_path

    def predict(self, encoded_data: EncodedData, label: Label):

        pass
        # predictions_proba = self.predict_proba(encoded_data, label)
        # return {label.name: [self.class_mapping[val] for val in (predictions_proba[label.name][:, 1] > 0.5).tolist()]}



    def predict_proba(self, encoded_data: EncodedData, label: Label):
        warnings.warn(f"{MotifClassifier.__name__}: cannot predict probabilities.")
        return None

    def fit(self, encoded_data: EncodedData, label: Label, cores_for_training: int = 2):
        train_data, validation_data = self._prepare_and_split_data(encoded_data)

        logging.info(f"{MotifClassifier.__name__}: finished training.")

    def fit_by_cross_validation(self, encoded_data: EncodedData, number_of_splits: int = 5, label: Label = None, cores_for_training: int = -1,
                                optimization_metric=None):
        logging.warning(f"{MotifClassifier.__name__}: cross_validation is not implemented for this method. Using standard fitting instead...")
        self.fit(encoded_data=encoded_data, label=label)


    def _prepare_and_split_data(self, encoded_data: EncodedData):
        train_indices, val_indices = Util.get_train_val_indices(len(encoded_data.example_ids), self.training_percentage)

        train_data = Util.subset_encoded_data(encoded_data, train_indices)
        val_data = Util.subset_encoded_data(encoded_data, val_indices)

        return train_data, val_data

    def store(self, path: Path, feature_names=None, details_path: Path = None):
        PathBuilder.build(path)
        #
        # torch.save(copy.deepcopy(self.CNN).state_dict(), str(path / "CNN.pt"))
        #
        # custom_vars = copy.deepcopy(vars(self))
        # del custom_vars["CNN"]
        # del custom_vars["result_path"]
        #
        # custom_vars["background_probabilities"] = custom_vars["background_probabilities"].tolist()
        # custom_vars["kernel_size"] = list(custom_vars["kernel_size"])
        # custom_vars["sequence_type"] = custom_vars["sequence_type"].name.lower()
        #
        # if self.label:
        #     custom_vars["label"] = vars(self.label)
        #
        # params_path = path / "custom_params.yaml"
        # with params_path.open('w') as file:
        #     yaml.dump(custom_vars, file)

    def load(self, path):
        pass
        # params_path = path / "custom_params.yaml"
        # with params_path.open("r") as file:
        #     custom_params = yaml.load(file, Loader=yaml.SafeLoader)
        #
        # for param, value in custom_params.items():
        #     if hasattr(self, param):
        #         if param == "label":
        #             setattr(self, "label", Label(**value))
        #         else:
        #             setattr(self, param, value)

        # self.background_probabilities = np.array(self.background_probabilities)
        # self.sequence_type = SequenceType[self.sequence_type.upper()]
        #
        # self._make_CNN()
        # self.CNN.load_state_dict(torch.load(str(path / "CNN.pt")))


    def check_if_exists(self, path):
        pass
        # return self.CNN is not None

    def get_params(self):
        pass
        # params = copy.deepcopy(vars(self))
        # params["CNN"] = copy.deepcopy(self.CNN).state_dict()
        # return params

    def get_label_name(self):
        return self.label.name

    def get_package_info(self) -> str:
        return Util.get_immuneML_version()

    def get_feature_names(self) -> list:
        return self.feature_names

    def can_predict_proba(self) -> bool:
        return False

    def get_class_mapping(self) -> dict:
        return self.class_mapping

    def get_compatible_encoders(self):
        from immuneML.encodings.motif_encoding.PositionalMotifEncoder import PositionalMotifEncoder
        return [PositionalMotifEncoder]

    def check_encoder_compatibility(self, encoder):
        """Checks whether the given encoder is compatible with this ML method, and throws an error if it is not."""
        is_valid = False

        # for encoder_class in self.get_compatible_encoders():
        #     if issubclass(encoder.__class__, encoder_class):
        #         is_valid = True
        #         break
        #
        # if not is_valid:
        #     raise ValueError(f"{encoder.__class__.__name__} is not compatible with ML Method {self.__class__.__name__}. "
        #                      f"Please use one of the following encoders instead: {', '.join([enc_class.__name__ for enc_class in self.get_compatible_encoders()])}")
        #
        # if (self.positional_channels == 3 and encoder.use_positional_info == False) or (self.positional_channels == 0 and encoder.use_positional_info == True):
        #     mssg = f"The specified parameters for {encoder.__class__.__name__} are not compatible with ML Method {self.__class__.__name__}. "
        #
        #     if encoder.use_positional_info:
        #         mssg += f"To include positional information, set the parameter 'positional_channels' of {self.__class__.__name__} to 3 (now {self.positional_channels}), " \
        #                 f"or to ignore positional information, set the parameter 'use_positional_info' of {encoder.__class__.__name__} to False (now {encoder.use_positional_info}). "
        #     else:
        #         mssg += f"To include positional information, set the parameter 'use_positional_info' of {encoder.__class__.__name__} to True (now {encoder.use_positional_info}), " \
        #                 f"or to ignore positional information, set the parameter 'positional_channels' of {self.__class__.__name__} to 0 (now {self.positional_channels})."
        #
        #     raise ValueError(mssg)
        #





