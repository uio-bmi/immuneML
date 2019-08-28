import pickle
from glob import glob

from source.IO.dataset_import.DataLoader import DataLoader
from source.IO.sequence_import.VDJdbSequenceImport import VDJdbSequenceImport
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset


class VDJDBLoader(DataLoader):
    """
    Loads data from VDJdb format into a Receptor- or SequenceDataset depending on the value of "paired" parameter
    """

    @staticmethod
    def load(path, params: dict = None) -> Dataset:

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
