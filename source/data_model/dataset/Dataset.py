# quality: gold

import uuid
from collections.abc import Iterable

import pandas as pd

from source.data_model.dataset.DatasetType import DatasetType
from source.data_model.encoded_data.EncodedData import EncodedData
from source.util.ReflectionHandler import ReflectionHandler


class Dataset:

    def __init__(self, data: Iterable = None, params: dict = None, encoded_data: EncodedData = None,
                 filenames: list = None, identifier: str = None, metadata_file: str = None,
                 dataset_type: DatasetType = DatasetType.REPERTOIRE):
        self.data = data
        self.params = params
        self.encoded_data = encoded_data
        self.id = identifier if identifier is not None else uuid.uuid1()
        self._filenames = sorted(filenames) if filenames is not None else []
        self.metadata_file = metadata_file
        self.type = dataset_type

    def add_data(self, data: Iterable):
        self.data = data

    def add_encoded_data(self, encoded_data: EncodedData):
        self.encoded_data = encoded_data

    def get_data(self, batch_size: int = 1):
        self._filenames.sort()
        return self._get_generator_class().build_generator(file_list=self._filenames, batch_size=batch_size)

    def _get_generator_class(self):
        return ReflectionHandler.get_class_by_name("{}Generator".format(self.type), "data_model/")

    def get_element(self, index: int = -1, identifier: str = ""):
        generator_class = self._get_generator_class()
        if index > -1:
            return generator_class.load_element_by_index(index)
        elif identifier != "":
            return generator_class.load_element_by_id(identifier)
        else:
            raise ValueError("Dataset: before calling get_element(), set index or identifier.")

    def get_element_count(self):
        return len(self._filenames)

    def set_filenames(self, filenames: list):
        self._filenames = sorted(filenames)

    def get_filenames(self):
        return self._filenames

    def get_metadata(self, field_names: list):
        df = pd.read_csv(self.metadata_file, sep=",", usecols=field_names)
        return df.to_dict("list")
