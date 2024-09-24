import copy
import logging
import shutil
import typing
from abc import ABC
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
from bionumpy import get_bufferclass_for_datatype

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.SequenceSet import Receptor, ReceptorSequence, AIRRSequenceSet, \
    build_dynamic_airr_sequence_set_dataclass, make_receptors_from_data, make_sequences_from_data
from immuneML.data_model.bnp_util import write_yaml, bnp_write_to_file, bnp_read_from_file, read_yaml, \
    extend_dataclass_with_dynamic_fields
from immuneML.data_model.datasets.Dataset import Dataset


@dataclass
class ElementDataset(Dataset, ABC):
    filename: Path = None
    element_count: int = None
    element_ids: list = None
    dataset_file: Path = None
    dynamic_fields: list = None
    bnp_dataclass: typing.Type = None
    _buffer_type = None

    def __post_init__(self):
        metadata = None
        if self.dynamic_fields is None:
            metadata = read_yaml(self.dataset_file)
            self.dynamic_fields = list(metadata['type_dict_dynamic_fields'].keys())
        if self.bnp_dataclass is None:
            if metadata is None:
                metadata = read_yaml(self.dataset_file)
                dynamic_fields = tuple(metadata['type_dict_dynamic_fields'].items())
            self.bnp_dataclass = extend_dataclass_with_dynamic_fields(AIRRSequenceSet, dynamic_fields)

    @property
    def buffer_type(self):
        if not self._buffer_type:
            self._buffer_type = get_bufferclass_for_datatype(self.bnp_dataclass, delimiter='\t', has_header=True)
        return self._buffer_type

    @property
    def data(self) -> AIRRSequenceSet:
        return bnp_read_from_file(self.filename, self.buffer_type, self.bnp_dataclass)

    def clone(self, keep_identifier: bool = False):
        dataset = self.__class__(labels=self.labels, encoded_data=copy.deepcopy(self.encoded_data),
                                 filename=self.filename, bnp_dataclass=self.bnp_dataclass,
                                 dataset_file=copy.deepcopy(self.dataset_file),
                                 name=self.name)
        if keep_identifier:
            dataset.identifier = self.identifier
        dataset.element_ids = self.element_ids
        return dataset


class SequenceDataset(ElementDataset):
    """A dataset class for sequence datasets (single chain). """

    @classmethod
    def build(cls, filename: Path, metadata_filename: Path, name: str = None):
        metadata = read_yaml(metadata_filename)
        dynamic_fields = metadata['type_dict_dynamic_fields']
        bnp_dc = extend_dataclass_with_dynamic_fields(AIRRSequenceSet,
                                                      tuple(dynamic_fields.items()))

        return SequenceDataset(name=name, filename=filename, dataset_file=metadata_filename,
                               dynamic_fields=list(dynamic_fields.keys()),
                               bnp_dataclass=bnp_dc)

    @classmethod
    def build_from_objects(cls, sequences: List[ReceptorSequence], path: Path, name: str = None,
                           labels: dict = None, region_type: RegionType = RegionType.IMGT_CDR3):
        name = name if name is not None else uuid4().hex
        filename = path / f"{name}.tsv"

        all_fields_dict = make_all_fields_dict_from_sequences(sequences, region_type)
        bnp_dc, type_dict = build_dynamic_airr_sequence_set_dataclass(all_fields_dict)
        bnp_write_to_file(filename, bnp_dc(**all_fields_dict))

        dataset_metadata = {
            'type_dict_dynamic_fields': {key: AIRRSequenceSet.TYPE_TO_STR[val] for key, val in type_dict.items()},
            'dataset_class': cls.__class__.__name__,
            'filename': filename}
        metadata_filename = path / f'dataset_{name}.yaml'
        write_yaml(metadata_filename, dataset_metadata)

        return SequenceDataset(filename=filename, name=name, labels=labels,
                               dynamic_fields=list(dataset_metadata['type_dict_dynamic_fields'].keys()),
                               dataset_file=metadata_filename, bnp_dataclass=bnp_dc)

    def get_metadata(self, field_names: list, return_df: bool = False):
        """Returns a dict or an equivalent pandas DataFrame with metadata information from Receptor objects for
        provided field names"""
        result = self.data.topandas()[field_names]

        return result if return_df else result.to_dict("list")

    def make_subset(self, example_indices, path, dataset_type: str):
        data = self.data[example_indices]
        name = f"subset_{self.name}"

        bnp_write_to_file(path / self.filename.name, data)

        metadata_filename = path / f'dataset_{name}.yaml'
        shutil.copyfile(self.dataset_file, metadata_filename)

        return SequenceDataset(filename=path / self.filename.name, name=name, labels=copy.deepcopy(self.labels),
                               dynamic_fields=self.dynamic_fields, dataset_file=metadata_filename,
                               bnp_dataclass=self.bnp_dataclass)

    def get_example_count(self):
        if self.element_count is None:
            self.element_count = len(self.data)
        return self.element_count

    def get_data(self, batch_size: int = 1, region_type: RegionType = RegionType.IMGT_CDR3):
        return make_sequences_from_data(self.data, self.dynamic_fields, region_type)

    def get_example_ids(self):
        return self.data.sequence_id

    def get_label_names(self):
        return list(self.labels.keys())

    def get_data_from_index_range(self, start_index: int, end_index: int):
        return self.data[start_index: end_index]


class ReceptorDataset(ElementDataset):
    """A dataset class for receptor datasets (paired chain)."""

    @classmethod
    def build(cls, filename: Path, metadata_filename: Path, name: str = None):
        metadata = read_yaml(metadata_filename)
        dynamic_fields = metadata['type_dict_dynamic_fields']
        bnp_dc = extend_dataclass_with_dynamic_fields(AIRRSequenceSet,
                                                      tuple(dynamic_fields.items()))

        return ReceptorDataset(name=name, filename=filename, dataset_file=metadata_filename,
                               dynamic_fields=list(dynamic_fields.keys()),
                               bnp_dataclass=bnp_dc)

    @classmethod
    def build_from_objects(cls, receptors: List[Receptor], path: Path, name: str = None,
                           labels: dict = None, region_type: RegionType = RegionType.IMGT_CDR3):
        name = name if name is not None else uuid4().hex
        filename = path / f"{name}.tsv"

        all_fields_dict = make_all_fields_dict_from_receptors(receptors, region_type)
        bnp_dc, type_dict = build_dynamic_airr_sequence_set_dataclass(all_fields_dict)
        bnp_write_to_file(filename, bnp_dc(**all_fields_dict))

        dataset_metadata = {
            'type_dict_dynamic_fields': {key: AIRRSequenceSet.TYPE_TO_STR[val] for key, val in type_dict.items()},
            'dataset_class': 'ReceptorDataset',
            'filename': filename}
        metadata_filename = path / f'dataset_{name}.yaml'
        write_yaml(metadata_filename, dataset_metadata)

        return ReceptorDataset(filename=filename, name=name, labels=labels,
                               dynamic_fields=list(dataset_metadata['type_dict_dynamic_fields'].keys()),
                               dataset_file=metadata_filename, bnp_dataclass=bnp_dc)

    def get_metadata(self, field_names: list, return_df: bool = False):
        """Returns a dict or an equivalent pandas DataFrame with metadata information from Receptor objects for
        provided field names"""
        result = self.data.topandas().groupby('cell_id', sort=False).agg(lambda x: "_".join([str(el) for el in list(set(x))])).reset_index()
        if any(field_name not in result.columns for field_name in field_names):
            logging.warning(f"ReceptorDataset {self.name}: not all requested metadata fields are available: {field_names}.")
        else:
            result = result[field_names]

        return result if return_df else result.to_dict('list')

    def make_subset(self, example_indices, path, dataset_type: str):
        true_indices = np.array([[ind, ind + 1] for ind in example_indices]).flatten()
        data = self.data[true_indices]
        name = f"subset_{self.name}"

        bnp_write_to_file(path / self.filename.name, data)

        metadata_filename = path / f'dataset_{name}.yaml'
        shutil.copyfile(self.dataset_file, metadata_filename)

        return ReceptorDataset(filename=path / self.filename.name, name=name, labels=copy.deepcopy(self.labels),
                               dynamic_fields=self.dynamic_fields, dataset_file=metadata_filename,
                               bnp_dataclass=self.bnp_dataclass)

    def get_example_count(self):
        if self.element_count is None:
            self.element_count = len(self.data) // 2
        return self.element_count

    def get_data(self, batch_size: int = 1):
        return make_receptors_from_data(self.data, self.dynamic_fields, f"ReceptorDataset {self.identifier}")

    def get_example_ids(self):
        return np.unique(self.data.cell_id).tolist()

    def get_label_names(self):
        return list(self.labels.keys())

    def get_data_from_index_range(self, start_index: int, end_index: int):
        return self.data[start_index * 2: end_index * 2]


def make_all_fields_dict_from_receptors(receptors: List[Receptor], region_type: RegionType = RegionType.IMGT_CDR3):
    all_fields = {seq_field.name: [] for seq_field in fields(AIRRSequenceSet)}
    field_types = {seq_field.name: seq_field.type for seq_field in fields(AIRRSequenceSet)}
    dynamic_fields = {}

    for index, receptor in enumerate(receptors):
        for seq_index, seq in enumerate([receptor.chain_1, receptor.chain_2]):
            for key in seq.metadata.keys():
                if key in all_fields:
                    all_fields[key].append(seq.metadata[key])
                elif key in dynamic_fields:
                    dynamic_fields[key].append(seq.metadata[key])
                else:
                    dynamic_fields[key] = [None for _ in range(index * 2 + seq_index)] + [seq.metadata[key]]
            for key in [f.name for f in fields(ReceptorSequence) if
                        f.name not in ['metadata', 'sequence_aa', 'sequence']]:
                all_fields[key].append(getattr(seq, key))
            all_fields[region_type.value].append(seq.sequence)
            all_fields[region_type.value + "_aa"].append(seq.sequence_aa)

    for field, values in all_fields.items():
        if all(val is None for val in values) or isinstance(values, list) and len(values) == 0:
            neutral_val = AIRRSequenceSet.get_neutral_value(field_types[field])
            all_fields[field] = [neutral_val for _ in range(len(receptors * 2))]

    return {**all_fields, **dynamic_fields}


def make_all_fields_dict_from_sequences(sequences: List[ReceptorSequence],
                                        region_type: RegionType = RegionType.IMGT_CDR3):
    airr_fields = fields(AIRRSequenceSet)
    all_fields = {seq_field.name: [] for seq_field in airr_fields}
    dynamic_fields = {}

    for index, sequence in enumerate(sequences):
        for key in sequence.metadata.keys():
            if key in all_fields:
                all_fields[key].append(sequence.metadata[key])
            elif key in dynamic_fields:
                dynamic_fields[key].append(sequence.metadata[key])
            else:
                dynamic_fields[key] = ['' for _ in range(index)] + [sequence.metadata[key]]
        for key in [f.name for f in fields(ReceptorSequence) if
                    f.name not in ['metadata', 'sequence_aa', 'sequence']]:
            all_fields[key].append(getattr(sequence, key))
        all_fields[region_type.value].append(sequence.sequence)
        all_fields[region_type.value + "_aa"].append(sequence.sequence_aa)

    for f in airr_fields:
        neutral_val = AIRRSequenceSet.get_neutral_value(f.type)
        if len(all_fields[f.name]) == 0:
            all_fields[f.name] = [neutral_val for _ in range(len(sequences))]
        elif any(val is None for val in all_fields[f.name]):
            all_fields[f.name] = [val if val is not None else neutral_val for val in all_fields[f.name]]

    return {**all_fields, **dynamic_fields}
