# quality: gold

import os
import pickle
from glob import glob

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset


class PickleImport(DataImport):
    """
    Imports the dataset from the pickle file which has previously been exported from immuneML. It does not perform any processing of
    examples in the dataset (i.e. repertoires), but relies on Repertoire objects that have been created previously.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_pickle_dataset: # user-defined dataset name
            format: Pickle
            params:
                path: path_to_dataset.iml_dataset # for datasets already in immuneML format
                metadata_file: metadata.csv # optional, but if specified, the dataset's metadata will be updated to this without changing Repertoire objects

    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        pickle_params = DatasetImportParams.build_object(**params)
        assert os.path.isfile(pickle_params.path), "PickleImport: the dataset file does not exist in the given path: " + pickle_params.path
        with open(pickle_params.path, "rb") as file:
            dataset = pickle.load(file)

        if pickle_params.metadata_file is not None and hasattr(dataset, "metadata_file"):
            dataset.metadata_file = pickle_params.metadata_file

        if isinstance(dataset, RepertoireDataset):
            dataset = PickleImport._update_repertoire_paths(pickle_params, dataset)

        return dataset

    @staticmethod
    def _update_repertoire_paths(pickle_params, dataset):
        path = PickleImport._discover_repertoire_path(pickle_params, dataset)
        if path is not None:
            for repertoire in dataset.repertoires:
                repertoire.data_filename = f"{path}{os.path.basename(repertoire.data_filename)}"
                repertoire.metadata_filename = f"{path}{os.path.basename(repertoire.metadata_filename)}"
        return dataset

    @staticmethod
    def _discover_repertoire_path(pickle_params, dataset):
        dataset_dir = os.path.dirname(pickle_params.path)
        if len(list(glob(f"{dataset_dir}/*.npy"))) == len(dataset.repertoires):
            path = dataset_dir + "/"
        elif len(list(glob(f"{dataset_dir}/repertoires/*.npy"))) == len(dataset.repertoires):
            path = dataset_dir + "/repertoires/"
        else:
            path = None

        return path
