# quality: gold

import collections
import uuid

from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.repertoire.RepertoireGenerator import RepertoireGenerator


class Dataset:

    def __init__(self, data: collections.Iterable = None, params: dict = None, encoded_data: EncodedData = None,
                 filenames: list = None, identifier: str = None):
        self.data = data
        self.params = params
        self.encoded_data = encoded_data
        self.filenames = sorted(filenames) if filenames is not None else []
        self.id = identifier if identifier is not None else uuid.uuid1()

    def add_data(self, data: collections.Iterable):
        self.data = data

    def add_encoded_data(self, encoded_data: EncodedData):
        self.encoded_data = encoded_data

    def get_data(self, batch_size: int = 1):
        sorted(self.filenames)
        return RepertoireGenerator.build_generator(file_list=self.filenames, batch_size=batch_size)

    def get_repertoire_count(self):
        return len(self.filenames)
