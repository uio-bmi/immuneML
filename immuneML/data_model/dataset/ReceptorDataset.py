import copy
import math
from pathlib import Path
from typing import List

import bionumpy as bnp
import pandas as pd

from immuneML.data_model.SequenceSet import SequenceSet
from immuneML.data_model.bnp_util import make_dynamic_seq_set_from_objs, bnp_write_to_file, \
    get_receptor_attributes_for_bnp, write_yaml
from immuneML.data_model.dataset.ElementDataset import ElementDataset
from immuneML.data_model.receptor.Receptor import Receptor


class ReceptorDataset(ElementDataset):
    """A dataset class for receptor datasets (paired chain). All the functionality is implemented in ElementDataset
    class, except creating a new dataset and obtaining metadata. """

    @classmethod
    def build_from_objects(cls, receptors: List[Receptor], file_size: int, path: Path, name: str = None,
                           labels: dict = None):

        file_count = math.ceil(len(receptors) / file_size)
        file_names = [
            path / f"batch{''.join(['0' for i in range(1, len(str(file_count)) - len(str(index)) + 1)])}{index}.tsv"
            for index in range(1, file_count + 1)]

        receptor_dc, types = make_dynamic_seq_set_from_objs(receptors)

        for index in range(file_count):
            field_vals = get_receptor_attributes_for_bnp(receptors[index * file_size:(index + 1) * file_size], receptor_dc, types)
            receptor_matrix = receptor_dc(**field_vals)
            bnp_write_to_file(file_names[index], receptor_matrix)

        dataset_metadata = {'type_dict': {key: SequenceSet.TYPE_TO_STR[val] for key, val in types.items()},
                            'element_class_name': type(receptors[0]).__name__,
                            'dataset_class': 'ReceptorDataset',
                            'filenames': [str(file) for file in file_names]}
        metadata_filename = path / f'dataset_{name}.yaml'
        write_yaml(metadata_filename, dataset_metadata)

        return ReceptorDataset(filenames=file_names, file_size=file_size, name=name, labels=labels,
                               element_class_name=type(receptors[0]).__name__ if len(receptors) > 0 else None,
                               dataset_file=metadata_filename,
                               buffer_type=bnp.io.delimited_buffers.get_bufferclass_for_datatype(receptor_dc,
                                                                                                 delimiter='\t',
                                                                                                 has_header=True))

    def get_metadata(self, field_names: list, return_df: bool = False):
        """Returns a dict or an equivalent pandas DataFrame with metadata information from Receptor objects for
        provided field names"""
        result = list(self.get_data(return_objects=False))[0]

        result = {field_name: getattr(result, field_name).tolist() for field_name in field_names}

        return pd.DataFrame(result) if return_df else result

    def clone(self, keep_identifier: bool = False):
        dataset = ReceptorDataset(self.labels, copy.deepcopy(self.encoded_data), copy.deepcopy(self.filenames),
                                  file_size=self.file_size, dataset_file=copy.deepcopy(self.dataset_file),
                                  name=self.name, element_class_name=self.element_generator.element_class_name)
        if keep_identifier:
            dataset.identifier = self.identifier
        dataset.element_ids = self.element_ids
        return dataset
