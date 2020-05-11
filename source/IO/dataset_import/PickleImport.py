# quality: gold

import os
import pickle

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.RepertoireDataset import RepertoireDataset


class PickleImport(DataImport):
    """
    Imports the dataset from the pickle file which has previously been exported from immuneML. It does not perform any processing of
    examples in the dataset (i.e. repertoires), but relies on Repertoire objects that have been created previously.

    Specification:
        path: path_to_dataset.pickle
        metadata_file: metadata.csv # if specified, the dataset's metadata will be updated to this without changing Repertoire objects
    """

    @staticmethod
    def import_dataset(params: dict) -> RepertoireDataset:
        pickle_params = DatasetImportParams.build_object(**params)
        assert os.path.isfile(pickle_params.path), "PickleImport: the dataset file does not exist in the given path: " + pickle_params.path
        with open(pickle_params.path, "rb") as file:
            dataset = pickle.load(file)

        if pickle_params.metadata_file is not None and hasattr(dataset, "metadata_file"):
            dataset.metadata_file = pickle_params.metadata_file

        return dataset
