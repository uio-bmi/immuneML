import copy
import logging
import math
from pathlib import Path
from typing import List

import bionumpy as bnp
import pandas as pd

from immuneML.data_model.SequenceSet import SequenceSet
from immuneML.data_model.bnp_util import bnp_write_to_file, make_dynamic_seq_set_from_objs, prepare_values_for_bnp, \
    write_yaml
from immuneML.data_model.dataset.ElementDataset import ElementDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.util.PathBuilder import PathBuilder


class SequenceDataset(ElementDataset):
    """A dataset class for sequence datasets (single chain). All the functionality is implemented in ElementDataset class, except creating a new
    dataset and obtaining metadata."""

    DEFAULT_FILE_SIZE = 100000

    @classmethod
    def build_from_objects(cls, sequences: List[ReceptorSequence], file_size: int, path: Path, name: str = None,
                           labels: dict = None):

        file_count = math.ceil(len(sequences) / file_size)
        PathBuilder.build(path)

        file_names = [
            path / f"batch{''.join(['0' for i in range(1, len(str(file_count)) - len(str(index)) + 1)])}{index}.tsv"
            for index in range(1, file_count + 1)]

        seq_set_dc, types = make_dynamic_seq_set_from_objs(sequences)

        for index in range(file_count):
            vals = {field_name: [seq.get_attribute(field_name) for seq in sequences[index * file_size:(index + 1) * file_size]]
                    for field_name in types.keys()}
            vals = prepare_values_for_bnp(vals, types)
            sequence_matrix = seq_set_dc(**vals)
            bnp_write_to_file(file_names[index], sequence_matrix)

        metadata = {
            'type_dict': {key: SequenceSet.TYPE_TO_STR[val] for key, val in types.items()},
            'dataset_class': 'SequenceDataset', 'element_class_name': ReceptorSequence.__name__,
            'filenames': [str(file) for file in file_names]
        }
        dataset_file = path / f'dataset_{name}.yaml'
        write_yaml(dataset_file, metadata)

        return SequenceDataset(filenames=file_names, file_size=file_size, name=name, labels=labels,
                               element_class_name=ReceptorSequence.__name__, dataset_file=dataset_file,
                               buffer_type=bnp.io.delimited_buffers.get_bufferclass_for_datatype(seq_set_dc,
                                                                                                 delimiter='\t',
                                                                                                 has_header=True)
                               )

    def __init__(self, **kwargs):
        super().__init__(**{**kwargs, **{'element_class_name': ReceptorSequence.__name__}})

    def get_metadata(self, field_names: list, return_df: bool = False):
        """Returns a dict or an equivalent pandas DataFrame with metadata information under 'custom_params' attribute in SequenceMetadata object
        for every sequence for provided field names"""
        result = {field: [] for field in field_names}
        for sequence in self.get_data():
            for field in field_names:
                result[field].append(sequence.metadata.get_attribute(field))

        for field in field_names:
            if all(item is None for item in result[field]):
                logging.warning(
                    f"{SequenceDataset.__name__}: none of the sequences in the dataset {self.name} have metadata field '{field}'. "
                    f"Returning 'None' instead...")
                result[field] = None

        return pd.DataFrame(result) if return_df else result

    def clone(self, keep_identifier: bool = False):
        dataset = SequenceDataset(labels=self.labels, encoded_data=copy.deepcopy(self.encoded_data),
                                  filenames=copy.deepcopy(self.filenames), dataset_file=self.dataset_file,
                                  file_size=self.file_size, name=self.name)
        if keep_identifier:
            dataset.identifier = self.identifier
        dataset.element_ids = self.element_ids
        return dataset
