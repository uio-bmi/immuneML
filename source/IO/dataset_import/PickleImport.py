# quality: gold

import os
import pickle
from glob import glob

import pandas as pd

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.ElementDataset import ElementDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants


class PickleImport(DataImport):
    """
    Imports the dataset from the pickle files previously exported by immuneML.
    PickleImport can import any kind of dataset (RepertoireDataset, SequenceDataset, ReceptorDataset).


    Arguments:

        path (str): The path to the previously created dataset file. This file should have an '.iml_dataset' extension.
        If the path has not been specified, immuneML attempts to load the dataset from a specified metadata file
        (only for RepertoireDatasets).

        metadata_file (str): An optional metadata file for a RepertoireDataset. If specified, the RepertoireDataset
        metadata will be updated to the newly specified metadata without otherwise changing the Repertoire objects


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_pickle_dataset:
            format: Pickle
            params:
                path: path/to/dataset.iml_dataset
                metadata_file: path/to/metadata.csv
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
        else:
            dataset = PickleImport._update_receptor_paths(pickle_params, dataset)

        return dataset

    @staticmethod
    def _import_from_path(pickle_params):
        with pickle_params.path.open("rb") as file:
            dataset = pickle.load(file)
        if pickle_params.metadata_file is not None and hasattr(dataset, "metadata_file"):
            dataset.metadata_file = pickle_params.metadata_file
        if hasattr(dataset, "metadata_file") and dataset.metadata_file is not None:
            metadata = pd.read_csv(dataset.metadata_file, comment=Constants.COMMENT_SIGN)
            metadata.to_csv(dataset.metadata_file, index=False)
        return dataset

    @staticmethod
    def _import_from_metadata(pickle_params,  dataset_name):
        assert False, "test this function" # todo test this
        with pickle_params.metadata_file.open("r") as file:
            dataset_filename = file.readline().replace(Constants.COMMENT_SIGN, "").replace("\n", "")
        pickle_params.path = f"{os.path.dirname(pickle_params.metadata_file)}/{dataset_filename}" \
            if os.path.dirname(pickle_params.metadata_file) != "" else dataset_filename

        assert os.path.isfile(pickle_params.path), f"PickleImport: dataset file {dataset_filename} specified in " \
                                                   f"{pickle_params.metadata_file} could not be found ({pickle_params.path} is not a file), " \
                                                   f"failed to import the dataset {dataset_name}."

        return PickleImport._import_from_path(pickle_params)

    @staticmethod
    def _update_repertoire_paths(pickle_params, dataset):
        path = PickleImport._discover_repertoire_path(pickle_params, dataset)
        if path is not None:
            for repertoire in dataset.repertoires:
                repertoire.data_filename = path / repertoire.data_filename.name
                repertoire.metadata_filename = path / repertoire.metadata_filename.name
        return dataset

    @staticmethod
    def _discover_dataset_dir(pickle_params):
        return pickle_params.path.parents[0]

    @staticmethod
    def _update_receptor_paths(pickle_params, dataset: ElementDataset):
        dataset_dir = PickleImport._discover_dataset_dir(pickle_params)

        if len(list(glob(f"{dataset_dir}*.pickle"))) == len(dataset.get_filenames()):
            path = dataset_dir
            new_filenames = []
            for file in dataset.get_filenames():
                new_filenames.append(f"{path}{os.path.basename(file)}")
            dataset.set_filenames(new_filenames)

        return dataset

    @staticmethod
    def _discover_repertoire_path(pickle_params, dataset):
        dataset_dir = PickleImport._discover_dataset_dir(pickle_params)

        if len(list(glob(f"{dataset_dir}*.npy"))) == len(dataset.repertoires):
            path = dataset_dir
        elif len(list(glob(f"{dataset_dir}repertoires/*.npy"))) == len(dataset.repertoires):
            path = dataset_dir + "repertoires/"
        else:
            path = None

        return path
