import os

import numpy as np
import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.preprocessing.Preprocessor import Preprocessor
from source.util.PathBuilder import PathBuilder


class PatientRepertoireCollector(Preprocessor):

    def __init__(self, result_path: str = None):
        self.result_path = result_path

    def process_dataset(self, dataset: RepertoireDataset, result_path: str = None):
        return PatientRepertoireCollector.process(dataset, {"result_path": result_path if result_path is not None else self.result_path})

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        rep_map = {}
        repertoires = []
        indices_to_keep = []

        processed_dataset = dataset.clone()
        PathBuilder.build(params["result_path"])

        for index, repertoire in enumerate(processed_dataset.get_data()):
            if repertoire.metadata["donor"] in rep_map.keys():
                sequences = np.append(repertoire.sequences, rep_map[repertoire.metadata["donor"]].sequences)
                del rep_map[repertoire.metadata["donor"]]
                repertoires.append(PatientRepertoireCollector.store_repertoire(
                    params["result_path"], repertoire, sequences))
            else:
                rep_map[repertoire.metadata["donor"]] = repertoire
                indices_to_keep.append(index)

        for key in rep_map.keys():
            repertoires.append(PatientRepertoireCollector.store_repertoire(params["result_path"], rep_map[key], rep_map[key].sequences))

        processed_dataset.repertoires = repertoires
        processed_dataset.metadata_file = PatientRepertoireCollector.build_new_metadata(dataset, indices_to_keep, params["result_path"])

        return processed_dataset

    @staticmethod
    def build_new_metadata(dataset, indices_to_keep, result_path: str):
        if dataset.metadata_file:
            df = pd.read_csv(dataset.metadata_file, index_col=0).iloc[indices_to_keep, :]
            path = result_path + "_{}_collected_repertoires.csv"\
                .format(os.path.splitext(os.path.basename(dataset.metadata_file))[0])
            df.to_csv(path)
        else:
            path = None
        return path

    @staticmethod
    def store_repertoire(path, repertoire, sequences):
        new_repertoire = Repertoire.build_from_sequence_objects(sequences, path, repertoire.metadata)
        return new_repertoire
