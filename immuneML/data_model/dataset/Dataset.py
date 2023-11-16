import abc
from typing import List


class Dataset:
    TRAIN = "train"
    TEST = "test"
    SUBSAMPLED = "subsampled"
    PREPROCESSED = "preprocessed"

    def __init__(self, encoded_data=None, name: str = None, identifier: str = None, labels: dict = None, example_weights: list = None):
        self.encoded_data = encoded_data
        self.identifier = identifier
        self.name = name if name is not None else self.identifier
        self.labels = labels
        self.example_weights = example_weights

    @classmethod
    @abc.abstractmethod
    def build_from_objects(cls, **kwargs):
        pass

    @abc.abstractmethod
    def make_subset(self, example_indices, path, dataset_type: str):
        pass

    @abc.abstractmethod
    def get_attribute(self, attribute: str, as_list: bool = True):
        pass

    @abc.abstractmethod
    def get_attributes(self, attributes: List[str], as_list: bool = True):
        pass

    @abc.abstractmethod
    def get_example_count(self):
        pass

    @abc.abstractmethod
    def get_data(self, batch_size: int = 1):
        pass

    @abc.abstractmethod
    def get_batch(self, batch_size: int = 1):
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
    def get_metadata(self, field_names: list, return_df: bool = False):
        pass

    @abc.abstractmethod
    def get_data_from_index_range(self, start_index: int, end_index: int):
        pass

    def set_example_weights(self, example_weights: list):
        if example_weights is not None:
            assert len(example_weights) == self.get_example_count(), f"{self.__class__.__name__}: trying to set example weights " \
                                                                 f"for dataset {self.identifier} but number of weights ({len(example_weights)}) " \
                                                                 f"does not match example count ({self.get_example_count()}). "
        self.example_weights = example_weights

    def get_example_weights(self):
        return self.example_weights