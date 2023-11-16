# quality: gold
import copy
import logging
import uuid
from pathlib import Path

import pandas as pd

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class RepertoireDataset(Dataset):

    @classmethod
    def build_from_objects(cls, **kwargs):
        ParameterValidator.assert_keys_present(list(kwargs.keys()), ['repertoires', 'path'], RepertoireDataset.__name__, RepertoireDataset.__name__)
        ParameterValidator.assert_all_type_and_value(kwargs['repertoires'], Repertoire, RepertoireDataset.__name__, 'repertoires')

        assert len(kwargs['repertoires']) > 0, "Cannot to construct a repertoire dataset without repertories."

        metadata_df = pd.DataFrame.from_records([{**rep.metadata, **{'filename': rep.data_filename}} for rep in kwargs['repertoires']])

        if 'field_list' in metadata_df.columns:
            metadata_df.drop(columns=['field_list'], inplace=True)

        metadata_path = PathBuilder.build(kwargs['path']) / 'metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)

        logging.info(f"Made new repertoire dataset at {kwargs['path']} with metadata at {metadata_path} "
                     f"with {len(kwargs['repertoires'])} repertoires.")

        dataset = RepertoireDataset(repertoires=kwargs['repertoires'], metadata_file=metadata_path)
        label_names = list(dataset.get_label_names(refresh=True))
        dataset.labels = {label: list(set(values)) for label, values in dataset.get_metadata(label_names).items()}

        return dataset

    @classmethod
    def build(cls, **kwargs):
        ParameterValidator.assert_keys_present(list(kwargs.keys()), ['metadata_file', 'name', 'repertoire_ids', 'metadata_fields'],
                                               RepertoireDataset.__name__, "repertoire dataset")
        repertoires = []
        metadata_df = pd.read_csv(kwargs['metadata_file'], comment=Constants.COMMENT_SIGN)
        for index, row in metadata_df.iterrows():
            filename = Path(kwargs['metadata_file']).parent / row['filename']
            if not filename.is_file() and 'repertoires' in str(filename):
                filename = filename.parent.parent / Path(row['filename']).name
            repertoire = Repertoire(data_filename=filename,
                                    metadata_filename=filename.parent / f'{filename.stem}_metadata.yaml',
                                    identifier=row['identifier'] if 'identifier' in row else uuid.uuid4().hex)
            repertoires.append(repertoire)

        if "repertoire_id" in kwargs.keys() and "repertoires" not in kwargs.keys() and kwargs['repertoire_id'] is not None:
            assert all(rep.identifier == kwargs['repertoire_id'][i] for i, rep in enumerate(repertoires)), \
                f"{RepertoireDataset.__name__}: repertoire ids from the iml_dataset file and metadata file don't match for the dataset " \
                f"{kwargs['name']} with identifier {kwargs['identifier']}."

        return RepertoireDataset(**{**kwargs, **{"repertoires": repertoires}})

    def __init__(self, labels: dict = None, encoded_data: EncodedData = None, repertoires: list = None, identifier: str = None,
                 metadata_file: Path = None, name: str = None, metadata_fields: list = None, repertoire_ids: list = None,
                 example_weights: list = None):
        super().__init__(encoded_data, name, identifier if identifier is not None else uuid.uuid4().hex, labels, example_weights)
        self.metadata_file = Path(metadata_file) if metadata_file is not None else None
        self.metadata_fields = metadata_fields
        self.repertoire_ids = repertoire_ids
        self.repertoires = repertoires

    def clone(self, keep_identifier: bool = False):
        dataset = RepertoireDataset(self.labels, copy.deepcopy(self.encoded_data), copy.deepcopy(self.repertoires),
                                    metadata_file=self.metadata_file, name=self.name)
        if keep_identifier:
            dataset.identifier = self.identifier
        return dataset

    def add_encoded_data(self, encoded_data: EncodedData):
        self.encoded_data = encoded_data

    def get_data(self, batch_size: int = 1):
        return self.repertoires

    def get_batch(self, batch_size: int = 1):
        return self.repertoires

    def get_repertoire(self, index: int = -1, repertoire_identifier: str = "") -> Repertoire:
        assert index != -1 or repertoire_identifier != "", \
            "RepertoireDataset: cannot import_dataset repertoire since the index nor identifier are set."
        return self.repertoires[index] if index != -1 else [rep for rep in self.repertoires if rep.identifier == repertoire_identifier][0]

    def get_example_count(self):
        return len(self.repertoires)

    def get_metadata_fields(self, refresh=False):
        """Returns the list of metadata fields, includes also the fields that will typically not be used as labels,
        like filename or identifier"""
        if self.metadata_fields is None or refresh:
            df = pd.read_csv(self.metadata_file, sep=",", nrows=0, comment=Constants.COMMENT_SIGN)
            self.metadata_fields = df.columns.values.tolist()
        return self.metadata_fields

    def get_label_names(self, refresh=False):
        """Returns the list of metadata fields which can be used as labels; if refresh=True, it reloads the fields
        from disk"""
        all_metadata_fields = set(self.get_metadata_fields(refresh))
        for non_label in ["subject_id", "filename", "repertoire_id", "identifier"]:
            if non_label in all_metadata_fields:
                all_metadata_fields.remove(non_label)

        return all_metadata_fields

    def get_metadata(self, field_names: list, return_df: bool = False):
        """
        A function to get the metadata of the repertoires. It can be useful in encodings or reports when the repertoire information needed is not
        present only in the label chosen for the ML model (e.g., disease), but also other information (e.g., age, HLA).

        Args:
            field_names (list): list of the metadata fields to return; the fields must be present in the metadata files. To find fields available, use :py:obj:`~immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset.get_label_names` function.
            return_df (bool): determines if the results should be returned as a dataframe where each column corresponds to a field or as a dictionary

        Returns:

            a dictionary where keys are fields names and values are lists of field values for each repertoire; alternatively returns the same information in dataframe format

        """
        assert isinstance(self.metadata_file, Path) and self.metadata_file.is_file(), \
            f"RepertoireDataset: for dataset {self.name} (id: {self.identifier}) metadata file is not set properly. The metadata file points to " \
            f"{self.metadata_file}."

        df = pd.read_csv(self.metadata_file, sep=",", usecols=field_names, comment=Constants.COMMENT_SIGN)
        if return_df:
            return df
        else:
            return df.to_dict("list")

    def get_filenames(self):
        """Returns the paths to files in which repertoire information is stored"""
        return [Path(filename) for filename in self.get_metadata(["filename"])["filename"]]

    def _build_new_metadata(self, indices, path: Path) -> Path:
        if self.metadata_file:
            df = pd.read_csv(self.metadata_file, comment=Constants.COMMENT_SIGN)
            df = df.iloc[indices, :]
            df.to_csv(path, index=False)
            return path
        else:
            return None

    def make_subset(self, example_indices, path: Path, dataset_type: str):
        """
        Creates a new dataset object with only those examples (repertoires) available which were given by index in example_indices argument.

        Args:
            example_indices (list): a list of indices of examples (repertoires) to use in the new dataset
            path (Path): a path where to store the newly created dataset
            dataset_type (str): a type of the dataset used as a part of the name of the resulting dataset; the values are defined as constants in :py:obj:`~immuneML.data_model.dataset.Dataset.Dataset`

        Returns:

            a new RepertoireDataset object which includes only the repertoires specified under example_indices

        """

        metadata_file = self._build_new_metadata(example_indices, path / f"{dataset_type}_metadata.csv")
        new_dataset = RepertoireDataset(repertoires=[self.repertoires[i] for i in example_indices], labels=copy.deepcopy(self.labels),
                                        metadata_file=metadata_file, identifier=str(uuid.uuid1()))

        original_example_weights = self.get_example_weights()
        if original_example_weights is not None:
            new_dataset.set_example_weights([original_example_weights[i] for i in example_indices])

        return new_dataset

    def get_repertoire_ids(self) -> list:
        """Returns a list of repertoire identifiers, same as get_example_ids()"""
        if self.repertoire_ids is None:
            self.repertoire_ids = [str(repertoire.identifier) for repertoire in self.repertoires]
        return self.repertoire_ids

    def get_example_ids(self):
        """Returns a list of example identifiers"""
        return self.get_repertoire_ids()

    def get_subject_ids(self):
        """Returns a list of subject identifiers"""
        return self.get_metadata(["subject_id"])["subject_id"]

    def get_data_from_index_range(self, start_index: int, end_index: int):
        return self.repertoires[start_index:end_index+1]
