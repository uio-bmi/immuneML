# quality: gold

import os
import pickle

from source.IO.dataset_import.DataLoader import DataLoader
from source.data_model.dataset.RepertoireDataset import RepertoireDataset


class PickleLoader(DataLoader):

    @staticmethod
    def load(path, params: dict = None) -> RepertoireDataset:
        assert(os.path.isfile(path)), "PickleLoader: the dataset file does not exist in the given path: " + path
        with open(path, "rb") as file:
            dataset = pickle.load(file)
        return dataset
