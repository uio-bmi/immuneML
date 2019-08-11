# quality: gold

import abc

from source.data_model.dataset.RepertoireDataset import RepertoireDataset


class DatasetEncoder(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def encode(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        pass

    @staticmethod
    @abc.abstractmethod
    def store(encoded_dataset: RepertoireDataset, params: dict):
        pass
