# quality: gold
import copy
import uuid

import pandas as pd
from pathlib import Path

from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.Constants import Constants


class RepertoireDataset(Dataset):

    def __init__(self, params: dict = None, encoded_data: EncodedData = None, repertoires: list = None, identifier: str = None,
                 metadata_file: Path = None, name: str = None):
        super().__init__()
        self.params = params
        self.encoded_data = encoded_data
        self.identifier = identifier if identifier is not None else uuid.uuid4().hex
        self.metadata_file = metadata_file
        self.metadata_fields = None
        self.repertoire_ids = None
        self.repertoires = repertoires
        self.name = name if name is not None else self.identifier

    def clone(self):
        return RepertoireDataset(self.params, copy.deepcopy(self.encoded_data), copy.deepcopy(self.repertoires),
                                 metadata_file=self.metadata_file)

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

    def get_metadata_fields(self):
        if self.metadata_fields is None:
            df = pd.read_csv(self.metadata_file, sep=",", nrows=0)
            self.metadata_fields = df.columns.values.tolist()
        return self.metadata_fields

    def get_metadata(self, field_names: list, return_df: bool = False):
        df = pd.read_csv(self.metadata_file, sep=",", usecols=field_names, comment=Constants.COMMENT_SIGN)
        if return_df:
            return df
        else:
            return df.to_dict("list")

    def _build_new_metadata(self, indices, path: Path) -> Path:
        if self.metadata_file:
            df = pd.read_csv(self.metadata_file, comment=Constants.COMMENT_SIGN)
            df = df.iloc[indices, :]
            df.to_csv(path, index=False)
            return path
        else:
            return None

    def make_subset(self, example_indices, path: Path, dataset_type: str):

        metadata_file = self._build_new_metadata(example_indices, path / f"{dataset_type}_metadata.csv")
        new_dataset = RepertoireDataset(repertoires=[self.repertoires[i] for i in example_indices], params=copy.deepcopy(self.params),
                                        metadata_file=metadata_file, identifier=str(uuid.uuid1()))

        return new_dataset

    def get_repertoire_ids(self) -> list:
        if self.repertoire_ids is None:
            self.repertoire_ids = [str(repertoire.identifier) for repertoire in self.repertoires]
        return self.repertoire_ids

    def get_example_ids(self):
        return self.get_repertoire_ids()
