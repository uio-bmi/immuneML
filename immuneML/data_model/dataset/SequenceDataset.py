import copy
import logging
import math
import pickle
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.data_model.dataset.ElementDataset import ElementDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class SequenceDataset(ElementDataset):
    """A dataset class for sequence datasets (single chain). All the functionality is implemented in ElementDataset class, except creating a new
    dataset and obtaining metadata."""

    @classmethod
    def build(cls, sequences: List[ReceptorSequence], file_size: int, path: Path, name: str = None):

        file_count = math.ceil(len(sequences) / file_size)
        file_names = [path / f"batch{''.join(['0' for i in range(1, len(str(file_count)) - len(str(index)) + 1)])}{index}.pickle"
                      for index in range(1, file_count+1)]

        for index in range(file_count):
            with file_names[index].open("wb") as file:
                pickle.dump(sequences[index*file_size:(index+1)*file_size], file)

        return SequenceDataset(filenames=file_names, file_size=file_size, name=name)

    def get_metadata(self, field_names: list, return_df: bool = False):
        """Returns a dict or an equivalent pandas DataFrame with metadata information under 'custom_params' attribute in SequenceMetadata object
        for every sequence for provided field names"""
        result = {field: [] for field in field_names}
        for sequence in self.get_data():
            for field in field_names:
                result[field].append(sequence.metadata.get_attribute(field))

        for field in field_names:
            if all(item is None for item in result[field]):
                logging.warning(f"{SequenceDataset.__name__}: none of the sequences in the dataset {self.name} have metadata field '{field}'. "
                                f"Returning 'None' instead...")
                result[field] = None

        return pd.DataFrame(result) if return_df else result

    def clone(self):
        return SequenceDataset(self.labels, copy.deepcopy(self.encoded_data), copy.deepcopy(self._filenames), file_size=self.file_size,
                               name=self.name)
