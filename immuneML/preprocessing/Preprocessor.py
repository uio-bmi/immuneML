import abc
from pathlib import Path

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset


class Preprocessor(metaclass=abc.ABCMeta):

    def __init__(self, result_path: Path = None):
        self.result_path = result_path

    @abc.abstractmethod
    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes: int = 1) -> RepertoireDataset:
        pass

    def check_dataset_type(self, dataset, valid_dataset_types: list, location: str):
        assert type(dataset) in valid_dataset_types, f"{location}: this preprocessing can only be applied to datasets of type: " \
                                                     f"{', '.join([dataset_type.__name__ for dataset_type in valid_dataset_types])}. " \
                                                     f"Your dataset is a {type(dataset).__name__}. " \
                                                     f"Please use a different preprocessing, or omit the preprocessing for this dataset."

    def keeps_example_count(self) -> bool:
        """
        Defines if the preprocessing can be run with TrainMLModel instruction; to be able to run with it, the preprocessing cannot change the
        number of examples in the dataset
        """
        return True
