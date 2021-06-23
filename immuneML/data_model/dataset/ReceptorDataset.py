import copy
import logging
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from immuneML.data_model.dataset.ElementDataset import ElementDataset
from immuneML.data_model.receptor.Receptor import Receptor


class ReceptorDataset(ElementDataset):
    """A dataset class for receptor datasets (paired chain). All the functionality is implemented in ElementDataset class, except creating a new
    dataset and obtaining metadata. """

    @classmethod
    def build(cls, **kwargs):
        return ReceptorDataset(**kwargs)

    @classmethod
    def build_from_objects(cls, receptors: List[Receptor], file_size: int, path: Path, name: str = None):

        file_count = math.ceil(len(receptors) / file_size)
        file_names = [path / f"batch{''.join(['0' for i in range(1, len(str(file_count)) - len(str(index)) + 1)])}{index}.npy"
                      for index in range(1, file_count + 1)]

        for index in range(file_count):
            receptor_matrix = np.core.records.fromrecords(
                [receptor.get_record() for receptor in receptors[index * file_size:(index + 1) * file_size]],
                names=type(receptors[0]).get_record_names())
            np.save(str(file_names[index]), receptor_matrix, allow_pickle=False)

        return ReceptorDataset(filenames=file_names, file_size=file_size, name=name,
                               element_class_name=type(receptors[0]).__name__ if len(receptors) > 0 else None)

    def get_metadata(self, field_names: list, return_df: bool = False):
        """Returns a dict or an equivalent pandas DataFrame with metadata information from Receptor objects for provided field names"""
        result = {field: [] for field in field_names}
        for receptor in self.get_data():
            for field in field_names:
                result[field].append(receptor.metadata[field] if receptor.metadata and field in receptor.metadata else None)

        for field in field_names:
            if all(item is None for item in result[field]):
                logging.warning(f"{ReceptorDataset.__name__}: none of the receptors in the dataset {self.name} have metadata field '{field}'. "
                                f"Returning 'None' instead...")
                result[field] = None

        return pd.DataFrame(result) if return_df else result

    def clone(self, keep_identifier: bool = False):
        dataset = ReceptorDataset(self.labels, copy.deepcopy(self.encoded_data), copy.deepcopy(self.filenames), file_size=self.file_size,
                                  name=self.name, element_class_name=self.element_generator.element_class_name)
        if keep_identifier:
            dataset.identifier = self.identifier
        dataset.element_ids = self.element_ids
        return dataset
