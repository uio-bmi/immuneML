import pickle
from pathlib import Path

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Label import Label
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.tool_interface import InterfaceController


class MLMethodTool(MLMethod):
    def __init__(self):
        super().__init__()

    def _start_subprocess(self):
        InterfaceController.check_running(self.name)

    def fit(self, encoded_data: EncodedData, label: Label, cores_for_training: int = 2):
        """
        # serialization of data
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
        self._start_subprocess()

        print("Pid: ", self.tool.pid)

        encoded_data_pickle = pickle.dumps(encoded_data)
        if self.tool.socket is None:
            self.tool.open_connection()
        self.tool.run_fit(encoded_data_pickle)

        print("fit is running in ml method")

    def predict(self, encoded_data: EncodedData, label: Label):
        encoded_data_pickle = pickle.dumps(encoded_data)
        result = self.tool.run_predict(encoded_data_pickle)
        return result

    def fit_by_cross_validation(self, encoded_data: EncodedData, number_of_splits: int = 5, label: Label = None,
                                cores_for_training: int = -1, optimization_metric=None):
        pass

    def store(self, path: Path, feature_names: list = None, details_path: Path = None):
        # TODO: subprocess call store
        # store(path, feature_names, details_path)
        print("TODO: store trained model in tool")

    def load(self, path: Path):
        pass

    def check_if_exists(self, path: Path) -> bool:
        pass

    def get_classes(self) -> list:
        return ["signal_disease"]

    def get_params(self):
        pass

    def predict_proba(self, encoded_data: EncodedData, Label: Label):
        encoded_data_pickle = pickle.dumps(encoded_data)
        result = self.tool.run_predict_proba(encoded_data_pickle)
        return result

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
