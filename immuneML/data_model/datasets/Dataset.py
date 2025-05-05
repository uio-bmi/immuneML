import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from immuneML.data_model.EncodedData import EncodedData


@dataclass
class Dataset:
    identifier: str = None
    name: str = None
    encoded_data: EncodedData = None
    labels: dict = field(default_factory=dict)
    dataset_file: Path = None

    TRAIN = "train"
    TEST = "test"
    SUBSAMPLED = "subsampled"
    PREPROCESSED = "preprocessed"

    def __post_init__(self):
        if self.name is None:
            self.name = self.identifier

    @classmethod
    @abc.abstractmethod
    def build_from_objects(cls, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def create_metadata_dict(cls, **kwargs):
        pass

    @abc.abstractmethod
    def make_subset(self, example_indices, path, dataset_type: str):
        pass

    @abc.abstractmethod
    def get_example_count(self):
        pass

    @abc.abstractmethod
    def get_data(self, batch_size: int = 1):
        pass

    @abc.abstractmethod
    def get_example_ids(self):
        pass

    @abc.abstractmethod
    def get_label_names(self):
        pass

    @abc.abstractmethod
    def clone(self, keep_identifier: bool = False):
        pass

    @abc.abstractmethod
    def get_metadata(self, field_names: Union[list, None], return_df: bool = False):
        pass

    @abc.abstractmethod
    def get_data_from_index_range(self, start_index: int, end_index: int):
        pass