# quality: gold

import logging
import pickle
from pathlib import Path

import pandas as pd

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ElementDataset import ElementDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants


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
        if hasattr(dataset, "metadata_file"):
            if pickle_params.metadata_file is not None:
                dataset.metadata_file = pickle_params.metadata_file
                metadata = pd.read_csv(dataset.metadata_file, comment=Constants.COMMENT_SIGN)
                metadata.to_csv(dataset.metadata_file, index=False)
            else:
                if dataset.metadata_file is not None and not dataset.metadata_file.is_file():
                    new_metadata_file = Path(dataset.metadata_file.name)
                    if new_metadata_file.is_file():
                        dataset.metadata_file = new_metadata_file
                        logging.warning(f"PickleImport: metadata file could not be found at {pickle_params.metadata_file}, "
                                        f"using {new_metadata_file} instead.")
                    else:
                        raise FileNotFoundError(f"PickleImport: the metadata file could not be found at {pickle_params.metadata_file}"
                                                f"or at {new_metadata_file}. Please update the path to the metadata file.")
        return dataset

    @staticmethod
    def _import_from_metadata(pickle_params,  dataset_name):
        with pickle_params.metadata_file.open("r") as file:
            dataset_filename = file.readline().replace(Constants.COMMENT_SIGN, "").replace("\n", "")
        pickle_params.path = pickle_params.metadata_file.parent / dataset_filename

        assert pickle_params.path.is_file(), f"PickleImport: dataset file {dataset_filename} specified in " \
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
        return pickle_params.path.parent

    @staticmethod
    def _update_receptor_paths(pickle_params, dataset: ElementDataset):
        dataset_dir = PickleImport._discover_dataset_dir(pickle_params)

        if len(list(dataset_dir.glob("*.pickle"))) == len(dataset.get_filenames()):
            path = dataset_dir
            new_filenames = []
            for file in dataset.get_filenames():
                new_filenames.append(path / file.name)
            dataset.set_filenames(new_filenames)

        return dataset

    @staticmethod
    def _discover_repertoire_path(pickle_params, dataset):
        dataset_dir = PickleImport._discover_dataset_dir(pickle_params)

        if len(list(dataset_dir.glob("*.npy"))) == len(dataset.repertoires):
            path = dataset_dir
        elif len(list(dataset_dir.glob("repertoires/*.npy"))) == len(dataset.repertoires):
            path = dataset_dir / "repertoires/"
        else:
            path = None

        return path
