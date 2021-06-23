# quality: gold

import logging
from pathlib import Path

import pandas as pd
import yaml

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ElementDataset import ElementDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.util.ReflectionHandler import ReflectionHandler


class ImmuneMLImport(DataImport):
    """
    Imports the dataset from the files previously exported by immuneML. It closely resembles AIRR format but relies on binary
    representations and is optimized for faster read-in at runtime.

    ImmuneMLImport can import any kind of dataset (RepertoireDataset, SequenceDataset, ReceptorDataset).

    This format includes:

     1. a dataset file in yaml format with iml_dataset extension with parameters:

        - name,
        - identifier,
        - metadata_file (for repertoire datasets),
        - metadata_fields (for repertoire datasets),
        - repertoire_ids (for repertoire datasets; for other dataset types this field is called element_ids),
        - labels,

     2. a csv metadata file (only for repertoire datasets),

     3. data files for different types of data. For repertoire datasets, data files include one binary numpy file per repertoire with sequences and
     associated information and one metadata yaml file per repertoire with details such as repertoire identifier, disease status, subject id and
     other similar available information. For sequence and receptor datasets, sequences or receptors respectively, are stored in batches in binary
     numpy files.

    Arguments:

        path (str): The path to the previously created dataset file. This file should have an '.iml_dataset' extension.
        If the path has not been specified, immuneML attempts to load the dataset from a specified metadata file
        (only for RepertoireDatasets).

        metadata_file (str): An optional metadata file for a RepertoireDataset. If specified, the RepertoireDataset
        metadata will be updated to the newly specified metadata without otherwise changing the Repertoire objects


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_dataset:
            format: ImmuneML
            params:
                path: path/to/dataset.iml_dataset
                metadata_file: path/to/metadata.csv

    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        iml_params = DatasetImportParams.build_object(**params)

        if iml_params.path is not None:
            dataset = ImmuneMLImport._import_from_path(iml_params)
        elif iml_params.metadata_file is not None:
            dataset = ImmuneMLImport._import_from_metadata(iml_params, dataset_name)
        else:
            raise ValueError(f"{ImmuneMLImport.__name__}: no path nor metadata file were defined under key {dataset_name}. At least one of these has "
                             f"to be specified to import the dataset.")

        if isinstance(dataset, RepertoireDataset):
            dataset = ImmuneMLImport._update_repertoire_paths(iml_params, dataset)
        else:
            dataset = ImmuneMLImport._update_receptor_paths(iml_params, dataset)

        return dataset

    @staticmethod
    def _import_dataset_dict(iml_params):
        with iml_params.path.open("r") as file:
            dataset_dict = yaml.safe_load(file)
        assert 'dataset_class' in dataset_dict, f"{ImmuneMLImport.__name__}: 'dataset_class' parameter is missing from the dataset file " \
                                                f"{iml_params.path}."
        dataset_class = ReflectionHandler.get_class_by_name(dataset_dict['dataset_class'])
        del dataset_dict['dataset_class']
        dataset = dataset_class.build(**dataset_dict)
        return dataset

    @staticmethod
    def _import_from_path(iml_params):
        dataset = ImmuneMLImport._import_dataset_dict(iml_params)
        if hasattr(dataset, "metadata_file"):
            if iml_params.metadata_file is not None:
                dataset.metadata_file = iml_params.metadata_file
                metadata = pd.read_csv(dataset.metadata_file, comment=Constants.COMMENT_SIGN)
                metadata.to_csv(dataset.metadata_file, index=False)
            else:
                if dataset.metadata_file is not None and not dataset.metadata_file.is_file():
                    new_metadata_file = Path(dataset.metadata_file.name)
                    if new_metadata_file.is_file():
                        dataset.metadata_file = new_metadata_file
                        logging.warning(f"{ImmuneMLImport.__name__}: metadata file could not be found at {iml_params.metadata_file}, "
                                        f"using {new_metadata_file} instead.")
                    else:
                        raise FileNotFoundError(f"{ImmuneMLImport.__name__}: the metadata file could not be found at {iml_params.metadata_file}"
                                                f"or at {new_metadata_file}. Please update the path to the metadata file.")
        return dataset

    @staticmethod
    def _import_from_metadata(iml_params, dataset_name):
        with iml_params.metadata_file.open("r") as file:
            dataset_filename = file.readline().replace(Constants.COMMENT_SIGN, "").replace("\n", "")
        iml_params.path = iml_params.metadata_file.parent / dataset_filename

        assert iml_params.path.is_file(), f"{ImmuneMLImport.__name__}: dataset file {dataset_filename} specified in " \
                                          f"{iml_params.metadata_file} could not be found ({iml_params.path} is not a file), " \
                                          f"failed to import the dataset {dataset_name}."

        return ImmuneMLImport._import_from_path(iml_params)

    @staticmethod
    def _update_repertoire_paths(iml_params, dataset):
        path = ImmuneMLImport._discover_repertoire_path(iml_params, dataset)
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
        dataset_dir = ImmuneMLImport._discover_dataset_dir(pickle_params)

        if len(list(dataset_dir.glob("*.npy"))) == len(dataset.get_filenames()):
            path = dataset_dir
            new_filenames = []
            for file in dataset.get_filenames():
                new_filenames.append(path / file.name)
            dataset.set_filenames(new_filenames)

        return dataset

    @staticmethod
    def _discover_repertoire_path(pickle_params, dataset):
        dataset_dir = ImmuneMLImport._discover_dataset_dir(pickle_params)

        if len(list(dataset_dir.glob("*.npy"))) == len(dataset.repertoires):
            path = dataset_dir
        elif len(list(dataset_dir.glob("repertoires/*.npy"))) == len(dataset.repertoires):
            path = dataset_dir / "repertoires/"
        else:
            path = None

        return path
