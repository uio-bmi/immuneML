from pathlib import Path
import json

import pandas
import pickle

from immuneML.dsl.ToolControllerML import ToolControllerML
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Label import Label
from scipy import sparse


class MLMethodTool(MLMethod):
    def __init__(self):
        super().__init__()
        self.tool = ToolControllerML()
        self.tool.start_subprocess()
        # self.tool.open_connection()
        # self.tool.run_fit()
        # self.tool.run_predict()
        # print("fit is running in ml method")

    def fit(self, encoded_data: EncodedData, label: Label, cores_for_training: int = 2):
        # TODO: subprocess call fit
        # self.tool.open_connection()
        """
        encoding = json.dumps(encoded_data.encoding)
        example_ids = json.dumps(encoded_data.example_ids)
        examples = pickle.dumps(encoded_data.examples)
        a = pickle.loads(examples)
        feature_annotations = encoded_data.feature_annotations.to_json()
        feature_names = json.dumps(encoded_data.feature_names)
        labels = json.dumps(encoded_data.labels)

        ab = pickle.dumps(encoded_data)

        encoded_data_json = {
            "encoding": encoding,
            "example_ids": example_ids,
            "examples": examples,
            "feature_annotations": feature_annotations,
            "labels": labels,

        }
        """
        encoded_data_pickle = pickle.dumps(encoded_data)
        self.tool.open_connection()
        self.tool.run_fit(encoded_data_pickle)

        print("fit is running in ml method")

    def predict(self, encoded_data: EncodedData, label: Label):
        # TODO: subprocess call predict

        encoded_data_pickle = pickle.dumps(encoded_data)
        self.tool.run_predict(encoded_data_pickle)

    def fit_by_cross_validation(self, encoded_data: EncodedData, number_of_splits: int = 5, label: Label = None,
                                cores_for_training: int = -1, optimization_metric=None):
        pass

    def store(self, path: Path, feature_names: list = None, details_path: Path = None):
        # TODO: subprocess call store
        # store(path, feature_names, details_path)
        pass

    def load(self, path: Path):
        pass

    def check_if_exists(self, path: Path) -> bool:
        pass

    def get_classes(self) -> list:
        pass

    def get_params(self):
        pass

    def predict_proba(self, encoded_data: EncodedData, Label: Label):
        # TODO: subprocess call predict_proba

        encoded_data_pickle = pickle.dumps(encoded_data)
        self.tool.run_predict_proba(encoded_data_pickle)

    def get_label_name(self) -> str:
        pass

    def get_package_info(self) -> str:
        pass

    def get_feature_names(self) -> list:
        pass

    def can_predict_proba(self) -> bool:
        pass

    def get_class_mapping(self) -> dict:
        pass

    def get_compatible_encoders(self):
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder

        return [KmerFrequencyEncoder]
