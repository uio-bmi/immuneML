import copy
import os
import pickle

import numpy as np
import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.preprocessing.Preprocessor import Preprocessor
from source.util.PathBuilder import PathBuilder


class PatientRepertoireCollector(Preprocessor):

    def __init__(self, result_path: str):
        self.result_path = result_path

    def process_dataset(self, dataset: RepertoireDataset):
        return PatientRepertoireCollector.process(dataset, {"result_path": self.result_path})

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        rep_map = {}
        filenames = []
        indices_to_keep = []

        processed_dataset = copy.deepcopy(dataset)
        PathBuilder.build(params["result_path"])

        for index, repertoire in enumerate(processed_dataset.get_data()):
            if repertoire.identifier in rep_map.keys():
                repertoire.sequences = np.append(repertoire.sequences, rep_map[repertoire.identifier].sequences)
                del rep_map[repertoire.identifier]
                filenames.append(PatientRepertoireCollector.store_repertoire(
                    params["result_path"] + repertoire.identifier + ".pkl", repertoire))
            else:
                rep_map[repertoire.identifier] = repertoire
                indices_to_keep.append(index)

        for key in rep_map.keys():
            filenames.append(PatientRepertoireCollector.store_repertoire(params["result_path"] + key + ".pkl",
                                                                         rep_map[key]))

        processed_dataset.set_filenames(filenames)
        processed_dataset.metadata_file = PatientRepertoireCollector.build_new_metadata(dataset, indices_to_keep)

        return processed_dataset

    @staticmethod
    def build_new_metadata(dataset, indices_to_keep):
        if dataset.metadata_file:
            df = pd.read_csv(dataset.metadata_file, index_col=0).iloc[indices_to_keep, :]
            path = os.path.dirname(os.path.abspath(dataset.metadata_file)) + "_{}_collected_repertoires.csv"\
                .format(os.path.splitext(os.path.basename(dataset.metadata_file))[0])
            df.to_csv(path)
        else:
            path = None
        return path

    @staticmethod
    def store_repertoire(path, repertoire):
        with open(path, "wb") as file:
            pickle.dump(repertoire, file)
        return path
