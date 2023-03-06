from dataclasses import dataclass
from pathlib import Path

import yaml

from immuneML.util.PathBuilder import PathBuilder


@dataclass
class MLMethodConfiguration:

    label_name: str = None
    label_values: list = None
    label_positive_class: str = None

    encoding_name: str = None
    encoding_class: str = None
    encoding_parameters: dict = None
    encoding_file: str = None

    preprocessing_sequence_name: str = None
    preprocessing_parameters: dict = None
    preprocessing_file: str = None

    ml_method: str = None
    ml_method_name: str = None

    train_dataset_id: str = None
    train_dataset_name: str = None
    software_used: str = None

    def store(self, path: Path):
        PathBuilder.build(path.parent)
        with path.open("w") as file:
            yaml.dump(self.__dict__, file)

    def load(self, path: Path):
        with path.open('r') as file:
            obj = yaml.safe_load(file)
        self.__init__(**obj)
