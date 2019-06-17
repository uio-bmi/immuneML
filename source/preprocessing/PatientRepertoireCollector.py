import copy
import os
import pickle

import numpy as np
import pandas as pd

from source.data_model.dataset.Dataset import Dataset
from source.preprocessing.Preprocessor import Preprocessor
from source.util.PathBuilder import PathBuilder


class PatientRepertoireCollector(Preprocessor):

    @staticmethod
    def process(dataset: Dataset, params: dict) -> Dataset:
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
        processed_dataset.metadata_path = PatientRepertoireCollector.build_new_metadata(dataset, indices_to_keep)

        return processed_dataset

    @staticmethod
    def build_new_metadata(dataset, indices_to_keep):
        if dataset.metadata_path:
            df = pd.read_csv(dataset.metadata_path, index_col=0).iloc[indices_to_keep, :]
            path = os.path.dirname(os.path.abspath(dataset.metadata_path)) + "_{}_collected_repertoires.csv"\
                .format(os.path.splitext(os.path.basename(dataset.metadata_path))[0])
            df.to_csv(path)
        else:
            path = None
        return path

    @staticmethod
    def store_repertoire(path, repertoire):
        with open(path, "wb") as file:
            pickle.dump(repertoire, file)
        return path
