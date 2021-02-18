import abc
from pathlib import Path

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset


class Preprocessor(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        pass

    @abc.abstractmethod
    def process_dataset(self, dataset: RepertoireDataset, result_path: Path) -> RepertoireDataset:
        pass

    @staticmethod
    def check_dataset_type(dataset, valid_dataset_types: list, location: str):
        assert type(dataset) in valid_dataset_types, f"{location}: this preprocessing can only be applied to datasets of type: {', '.join([dataset_type.__name__ for dataset_type in valid_dataset_types])}. " \
                                                     f"Your dataset is a {type(dataset).__name__}. " \
                                                     f"Please use a different preprocessing, or omit the preprocessing for this dataset."