# quality: gold

import abc

from immuneML.data_model.dataset.Dataset import Dataset


class DataImport(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def import_dataset(params, dataset_name: str) -> Dataset:
        pass

