import pickle
from glob import glob

import pandas as pd

from source.IO.dataset_import.DataLoader import DataLoader
from source.IO.sequence_import.VDJdbSequenceImport import VDJdbSequenceImport
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.util.PathBuilder import PathBuilder


class VDJDBLoader(DataLoader):
    """
    Loads data from VDJdb format into a Receptor- or SequenceDataset depending on the value of "paired" parameter or
    to RepertoireDataset (consisting of a list of receptor sequences)
    """

    @staticmethod
    def load(path: str = "", params: dict = None) -> Dataset:

        PathBuilder.build(params["result_path"])

        if "metadata_file" in params and "metadata_file" is not None:
            dataset = VDJDBLoader.load_repertoire_dataset(params)
        else:
            dataset = VDJDBLoader.load_sequence_dataset(path, params)
        return dataset

    @staticmethod
    def load_repertoire_dataset(params: dict) -> Dataset:
        metadata = pd.read_csv(params["metadata_file"])
        labels = {key: set() for key in metadata.keys() if key != "filename"}
        repertoires = []

        PathBuilder.build(params["result_path"])
        for index, row in metadata.iterrows():
            repertoire = VDJDBLoader.load_repertoire(index, row, params["result_path"])

            for key in labels.keys():
                labels[key].add(row[key])

            repertoires.append(repertoire)

        return RepertoireDataset(params=labels, repertoires=repertoires, metadata_file=params["metadata_file"])

    @staticmethod
    def store_repertoire(repertoire, params):
        filename = params["result_path"] + repertoire.identifier + ".pkl"
        with open(filename, "wb") as file:
            pickle.dump(repertoire, file)
        return filename

    @staticmethod
    def load_repertoire(index: int, row, result_path: str):
        sequences = VDJdbSequenceImport.import_items(row["filename"])
        return Repertoire.build_from_sequence_objects(sequences, result_path,
                                                      {key: row[key] for key in row.keys() if key != "filename"})

    @staticmethod
    def load_sequence_dataset(path: str, params: dict) -> Dataset:

        filenames = glob(path + "*.tsv", recursive=params["recursive"])
        file_index = 0
        dataset_filenames = []

        for index, filename in enumerate(filenames):
            items = VDJdbSequenceImport.import_items(filename, paired=params["paired"])

            while len(items) > params["file_size"] or (index == len(filenames)-1 and len(items) > 0):
                dataset_filenames.append(params["result_path"] + "batch_{}.pickle".format(file_index))
                VDJDBLoader.store_items(dataset_filenames, items, params["file_size"])
                items = items[params["file_size"]:]
                file_index += 1

        return ReceptorDataset(filenames=dataset_filenames, file_size=params["file_size"]) if params["paired"] \
            else SequenceDataset(filenames=dataset_filenames, file_size=params["file_size"])

    @staticmethod
    def store_items(dataset_filenames: list, items: list, file_size: int):
        with open(dataset_filenames[-1], "wb") as file:
            pickle.dump(items[:file_size], file)
