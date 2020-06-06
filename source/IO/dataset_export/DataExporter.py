# quality: gold

import abc

from source.data_model.dataset.Dataset import Dataset


class DataExporter(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def export(dataset: Dataset, path):
        pass
