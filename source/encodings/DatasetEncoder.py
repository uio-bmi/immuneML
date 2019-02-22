# quality: gold

import abc

from source.data_model.dataset.Dataset import Dataset


class DatasetEncoder(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def encode(dataset: Dataset, params: dict) -> Dataset:
        pass

    @staticmethod
    @abc.abstractmethod
    def store(encoded_dataset: Dataset, params: dict):
        pass
