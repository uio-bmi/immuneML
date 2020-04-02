import pickle
from glob import glob

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.IO.sequence_import.VDJdbSequenceImport import VDJdbSequenceImport
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.util.ImportHelper import ImportHelper


class VDJDBImport(DataImport):
    """
    Imports data from VDJdb format into a ReceptorDataset or SequenceDataset depending on the value of "paired" parameter or
    to RepertoireDataset (a set of repertoires consisting of a list of receptor sequences).


    """

    @staticmethod
    def import_dataset(params: dict) -> Dataset:
        vdjdb_params = DatasetImportParams.build_object(**params)
        if vdjdb_params.metadata_file is not None:
            dataset = VDJDBImport.load_repertoire_dataset(vdjdb_params)
        else:
            dataset = VDJDBImport.load_sequence_dataset(vdjdb_params)
        return dataset

    @staticmethod
    def load_repertoire_dataset(params: DatasetImportParams) -> Dataset:
        return ImportHelper.import_repertoire_dataset(VDJDBImport.preprocess_repertoire, params)

    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams) -> dict:
        return ImportHelper.load_repertoire_as_dataframe(metadata, params)

    @staticmethod
    def load_sequence_dataset(params: DatasetImportParams) -> Dataset:

        filenames = glob(params.path + "*.tsv")
        file_index = 0
        dataset_filenames = []

        for index, filename in enumerate(filenames):
            items = VDJdbSequenceImport.import_items(filename, paired=params.paired)

            while len(items) > params.file_size or (index == len(filenames)-1 and len(items) > 0):
                dataset_filenames.append(params.result_path + "batch_{}.pickle".format(file_index))
                VDJDBImport.store_items(dataset_filenames, items, params.file_size)
                items = items[params.file_size:]
                file_index += 1

        return ReceptorDataset(filenames=dataset_filenames, file_size=params.file_size) if params.paired \
            else SequenceDataset(filenames=dataset_filenames, file_size=params.file_size)

    @staticmethod
    def store_items(dataset_filenames: list, items: list, file_size: int):
        with open(dataset_filenames[-1], "wb") as file:
            pickle.dump(items[:file_size], file)
