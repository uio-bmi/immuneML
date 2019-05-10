import copy
import pickle

import numpy as np

from source.data_model.dataset.Dataset import Dataset
from source.preprocessing.Preprocessor import Preprocessor
from source.util.PathBuilder import PathBuilder


class PatientRepertoireCollector(Preprocessor):

    @staticmethod
    def process(dataset: Dataset, params: dict) -> Dataset:
        rep_map = {}
        filenames = []

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

        for key in rep_map.keys():
            filenames.append(PatientRepertoireCollector.store_repertoire(params["result_path"] + key + ".pkl",
                                                                         rep_map[key]))

        processed_dataset.filenames = filenames

        return processed_dataset

    @staticmethod
    def store_repertoire(path, repertoire):
        with open(path, "wb") as file:
            pickle.dump(repertoire, file)
        return path
