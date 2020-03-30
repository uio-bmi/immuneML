# quality: gold

import abc

from source.data_model.dataset.Dataset import Dataset


class DataImport(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def import_dataset(params) -> Dataset:
        pass
