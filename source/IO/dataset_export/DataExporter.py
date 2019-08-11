# quality: gold

import abc

from source.data_model.dataset.RepertoireDataset import RepertoireDataset


class DataExporter(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def export(dataset: RepertoireDataset, path, filename):
        pass
