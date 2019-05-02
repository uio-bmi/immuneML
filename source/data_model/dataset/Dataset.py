# quality: gold

import collections
import uuid
from collections import Iterable

import numpy as np

from source.data_model.repertoire.RepertoireGenerator import RepertoireGenerator


class Dataset:

    def __init__(self, data: collections.Iterable = None, params: dict = None, encoded_data=None, filenames: list = None, identifier: str = None):
        self.data = data
        self.params = params
        self.encoded_data = encoded_data
        self.filenames = sorted(filenames) if filenames is not None else []
        self.id = identifier if identifier is not None else uuid.uuid1()

    def add_data(self, data: collections.Iterable):
        self.data = data

    def add_encoded_data(self, encoded_data: dict):
        assert "repertoires" in encoded_data and isinstance(encoded_data["repertoires"], Iterable), "Repertoires are not properly specified when adding encoded_data to the dataset object."
        assert "labels" in encoded_data, "Labels are not properly specified when adding encoded data to the dataset object."

        self.encoded_data = encoded_data

    def get_data(self, batch_size: int = 1):
        sorted(self.filenames)
        return RepertoireGenerator.build_generator(file_list=self.filenames, batch_size=batch_size)

    def get_repertoire_count(self):
        return len(self.filenames)
