# quality: gold

import os
import pickle
from glob import glob

import pandas as pd

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants


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

        if pickle_params.path is not None:
            dataset = PickleImport._import_from_path(pickle_params)
        elif pickle_params.metadata_file is not None:
            dataset = PickleImport._import_from_metadata(pickle_params, dataset_name)
        else:
            raise ValueError(f"PickleImport: no path nor metadata file were defined under key {dataset_name}. At least one of these has "
                             f"to be specified to import the dataset.")

        if isinstance(dataset, RepertoireDataset):
            dataset = PickleImport._update_repertoire_paths(pickle_params, dataset)

        return dataset

    @staticmethod
    def _import_from_path(pickle_params):
        with open(pickle_params.path, "rb") as file:
            dataset = pickle.load(file)
        if pickle_params.metadata_file is not None and hasattr(dataset, "metadata_file"):
            dataset.metadata_file = pickle_params.metadata_file
        if hasattr(dataset, "metadata_file") and dataset.metadata_file is not None:
            metadata = pd.read_csv(dataset.metadata_file, comment=Constants.COMMENT_SIGN)
            metadata.to_csv(dataset.metadata_file, index=False)
        return dataset

    @staticmethod
    def _import_from_metadata(pickle_params,  dataset_name):
        with open(pickle_params.metadata_file, "r") as file:
            dataset_filename = file.readline().replace(Constants.COMMENT_SIGN, "")
        pickle_params.path = f"{os.path.dirname(pickle_params.metadata_file)}/{dataset_filename}"

        assert os.path.isfile(pickle_params.path), f"PickleImport: dataset file {dataset_filename} specified in " \
                                                   f"{pickle_params.metadata_file} could not be found, failed to import the dataset " \
                                                   f"{dataset_name}."

        return PickleImport._import_from_path(pickle_params)

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
