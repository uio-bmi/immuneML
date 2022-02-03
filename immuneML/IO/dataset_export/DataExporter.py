# quality: gold

import abc
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.receptor.RegionType import RegionType


class DataExporter(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def export(dataset: Dataset, path: Path, region_type: RegionType = None, number_of_processes: int = 1):
        pass
