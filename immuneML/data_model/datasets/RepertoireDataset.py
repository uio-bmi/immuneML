import copy
import logging
from collections import ChainMap
from datetime import datetime
import logging
import uuid
from pathlib import Path
from uuid import uuid4

import pandas as pd

from immuneML import Constants
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.data_model.bnp_util import write_yaml, write_dataset_yaml
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class RepertoireDataset(Dataset):

    @classmethod
    def build_from_objects(cls, **kwargs):
        ParameterValidator.assert_keys_present(list(kwargs.keys()), ['repertoires', 'path'], RepertoireDataset.__name__,
                                               RepertoireDataset.__name__)
        ParameterValidator.assert_all_type_and_value(kwargs['repertoires'], Repertoire, RepertoireDataset.__name__,
                                                     'repertoires')

        assert len(kwargs['repertoires']) > 0, "Cannot to construct a repertoire dataset without repertories."

        metadata_df = pd.DataFrame.from_records(
            [{**rep.metadata, **{'filename': rep.data_filename.name}} for rep in kwargs['repertoires']])

        if 'field_list' in metadata_df.columns:
            metadata_df.drop(columns=['field_list'], inplace=True)

        metadata_path = PathBuilder.build(kwargs['path']) / 'metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)

        logging.info(f"Made new repertoire dataset at {kwargs['path']} with metadata at {metadata_path} "
                     f"with {len(kwargs['repertoires'])} repertoires.")

        name = kwargs.get('name', None)

        dataset = RepertoireDataset(repertoires=kwargs['repertoires'], metadata_file=metadata_path, name=name)
        label_names = list(dataset.get_label_names(refresh=True))
        dataset.labels = {label: list(set(values)) for label, values in dataset.get_metadata(label_names).items()}

        dataset_file = PathBuilder.build(kwargs['path']) / 'dataset.yaml'
        dataset_meta_content = cls.create_metadata_dict(labels=dataset.labels,
                                                        identifier=dataset.identifier,
                                                        name=dataset.name,
                                                        metadata_file=str(metadata_path.name))

        write_dataset_yaml(dataset_file, dataset_meta_content)
        dataset.dataset_file = dataset_file

        return dataset

    @classmethod
    def build(cls, **kwargs):
        ParameterValidator.assert_keys_present(list(kwargs.keys()),
                                               ['metadata_file', 'name', 'repertoire_ids', 'metadata_fields'],
                                               RepertoireDataset.__name__, "repertoire dataset")
        repertoires = []
        metadata_df = pd.read_csv(kwargs['metadata_file'], comment=Constants.COMMENT_SIGN)
        for index, row in metadata_df.iterrows():
            filename = Path(kwargs['metadata_file']).parent / row['filename']
            if not filename.is_file() and 'repertoires' in str(filename):
                filename = filename.parent.parent / Path(row['filename']).name
            repertoire = Repertoire(data_filename=filename,
                                    metadata_filename=filename.parent / f'{filename.stem}_metadata.yaml',
                                    identifier=row['identifier'] if 'identifier' in row else uuid4().hex)
            repertoires.append(repertoire)

        if "repertoire_id" in kwargs.keys() and "repertoires" not in kwargs.keys() and kwargs[
            'repertoire_id'] is not None:
            assert all(rep.identifier == kwargs['repertoire_id'][i] for i, rep in enumerate(repertoires)), \
                f"{RepertoireDataset.__name__}: repertoire ids from the dataset file and metadata file don't match for the dataset " \
                f"{kwargs['name']} with identifier {kwargs['identifier']}."

        return RepertoireDataset(**{**kwargs, **{"repertoires": repertoires}})

    @classmethod
    def create_metadata_dict(cls, metadata_file, labels, name, identifier=None):
         return {"metadata_file": Path(metadata_file).name,
                 # "type_dict_dynamic_fields": type_dict,
                 "labels": {} if labels is None else labels,
                 "name": name,
                 "identifier": identifier if identifier is not None else uuid4().hex,
                 "dataset_type": cls.__name__,
                 "timestamp": datetime.now()}

    def __init__(self, labels: dict = None, encoded_data: EncodedData = None, repertoires: list = None,
                 identifier: str = None, metadata_file: Path = None, name: str = None, metadata_fields: list = None,
                 repertoire_ids: list = None, dataset_file: Path = None):
        super().__init__(encoded_data=encoded_data, name=name,
                         identifier=identifier if identifier is not None else uuid4().hex, labels=labels)
        self.metadata_file = Path(metadata_file) if metadata_file is not None else None
        self.metadata_fields = metadata_fields
        self.repertoire_ids = repertoire_ids
        self.repertoires = repertoires
        self.dataset_file = dataset_file

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

    def get_repertoire(self, index: int = -1, repertoire_identifier: str = "") -> Repertoire:
        assert index != -1 or repertoire_identifier != "", \
            "RepertoireDataset: cannot get repertoire since the index nor identifier are set."
        return self.repertoires[index] if index != -1 else \
            [rep for rep in self.repertoires if rep.identifier == repertoire_identifier][0]

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
        for non_label in ["subject_id", "filename", "repertoire_id", "identifier", "type_dict_dynamic_fields", "dataset_type"]:
            if non_label in all_metadata_fields:
                all_metadata_fields.remove(non_label)

        return all_metadata_fields

    def get_metadata(self, field_names: list, return_df: bool = False):
        assert isinstance(self.metadata_file, Path) and self.metadata_file.is_file(), \
            (f"RepertoireDataset: for dataset {self.name} (id: {self.identifier}) metadata file is not set properly. "
             f"The metadata file points to {self.metadata_file}.")

        df = pd.read_csv(self.metadata_file, sep=",", usecols=field_names, comment=Constants.COMMENT_SIGN)
        if return_df:
            return df
        else:
            return df.to_dict("list")

    def get_filenames(self):
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
        new_dataset = RepertoireDataset(repertoires=[self.repertoires[i] for i in example_indices],
                                        labels=copy.deepcopy(self.labels),
                                        metadata_file=metadata_file, identifier=str(uuid4()))

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
        return self.repertoires[start_index:end_index + 1]
