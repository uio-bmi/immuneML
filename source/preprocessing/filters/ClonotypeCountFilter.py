import copy
import os

import pandas as pd

from source.data_model.dataset.Dataset import Dataset
from source.preprocessing.Preprocessor import Preprocessor


class ClonotypeCountFilter(Preprocessor):

    def __init__(self, lower_limit: int = -1, upper_limit: int = -1):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def process_dataset(self, dataset: Dataset):
        params = {}
        if self.lower_limit > -1:
            params["lower_limit"] = self.lower_limit
        if self.upper_limit > -1:
            params["upper_limit"] = self.upper_limit
        return ClonotypeCountFilter.process(dataset, params)

    @staticmethod
    def process(dataset: Dataset, params: dict) -> Dataset:
        processed_dataset = copy.deepcopy(dataset)
        filenames = []
        indices =[]
        for index, repertoire in enumerate(dataset.get_data()):
            if "lower_limit" in params.keys() and len(repertoire.sequences) >= params["lower_limit"] or \
                 "upper_limit" in params.keys() and len(repertoire.sequences) <= params["upper_limit"]:
                filenames.append(dataset.get_filenames()[index])
                indices.append(index)
        processed_dataset.set_filenames(filenames)
        processed_dataset.metadata_file = ClonotypeCountFilter.build_new_metadata(dataset, indices)
        return processed_dataset

    @staticmethod
    def build_new_metadata(dataset, indices_to_keep):
        if dataset.metadata_file:
            df = pd.read_csv(dataset.metadata_file, index_col=0).iloc[indices_to_keep, :]
            path = os.path.dirname(os.path.abspath(dataset.metadata_file)) + "/_{}_clonotype_count_filtered.csv"\
                .format(os.path.splitext(os.path.basename(dataset.metadata_file))[0])
            df.to_csv(path)
        else:
            path = None
        return path
