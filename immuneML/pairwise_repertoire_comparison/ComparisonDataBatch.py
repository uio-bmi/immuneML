import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from immuneML.util.PathBuilder import PathBuilder


@dataclass
class ComparisonDataBatch:
    """
    Arguments:

        matrix: array with dimension items x repertoires, where items are defined by comparison attributes specified in ComparisonData
                class and can include, for instance, receptor sequences or combinations of receptor sequences and V and J gene

        items: the item names extracted from the repertoires in the dataset on which the repertoires are evaluated (e.g. sequences or
                combinations of sequences and genes

        repertoire_index_mapping: a mapping between the repertoire identifier (a string) and a column number for faster access of columns
                (repertoire vectors w.r.t. given items) in the comparison data matrix where columns correspond to repertoires

        path (Path): path to directory where comp data is stored

        identifier (int): identifier of the batch

    """

    items: list
    repertoire_index_mapping: Dict[str, int]
    path: Path
    identifier: int
    matrix: np.ndarray = None

    def store(self):
        PathBuilder.build(self.path)
        np.save(self.path / f"{self.identifier}.npy", self.matrix)

        np.save(self.path / f"{self.identifier}_items.npy", self.items)

        batch_vars = vars(self)
        del batch_vars["matrix"]
        del batch_vars["items"]

        pkl_path = self.path / f"{self.identifier}.pkl"
        with pkl_path.open("wb") as file:
            pickle.dump(batch_vars, file)

    def load(self):
        file_path = self.path / f'{self.identifier}.pkl'
        if file_path.is_file():
            with file_path.open('rb') as file:
                batch_vars = pickle.load(file)

            for v in batch_vars:
                if hasattr(self, v):
                    setattr(self, v, batch_vars[v])
        else:
            logging.warning(f"ComparisonDataBatch: path {file_path} does not exist, returning the same object...")

        return self

    def get_items(self):
        if self.matrix is None:
            return np.load(self.path / f"{self.identifier}_items.npy", allow_pickle=True)
        else:
            return self.items

    def get_matrix(self):
        if self.matrix is None:
            return np.load(self.path / f"{self.identifier}.npy", allow_pickle=True)
        else:
            return self.matrix
