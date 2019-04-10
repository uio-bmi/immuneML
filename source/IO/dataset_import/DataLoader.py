# quality: gold

import abc

from source.data_model.dataset.Dataset import Dataset


class DataLoader(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def load(path, params: dict = None) -> Dataset:
        pass
