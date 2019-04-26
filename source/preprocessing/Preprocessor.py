import abc

from source.data_model.dataset.Dataset import Dataset


class Preprocessor(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def process(dataset: Dataset, params: dict) -> Dataset:
        pass