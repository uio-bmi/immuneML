import copy
import itertools
import logging
import shutil
import typing
from datetime import datetime
from abc import ABC
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
import pandas as pd
from bionumpy import get_bufferclass_for_datatype

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.SequenceSet import Receptor, ReceptorSequence, AIRRSequenceSet, \
    build_dynamic_airr_sequence_set_dataclass, make_receptors_from_data, make_sequences_from_data
from immuneML.data_model.bnp_util import bnp_write_to_file, bnp_read_from_file, read_yaml, \
    extend_dataclass_with_dynamic_fields, write_dataset_yaml, make_full_airr_seq_set_df
from immuneML.data_model.datasets.Dataset import Dataset


@dataclass
class ElementDataset(Dataset, ABC):
    filename: Path = None
    element_count: int = None
    element_ids: list = None
    dataset_file: Path = None
    dynamic_fields: dict = None
    bnp_dataclass: typing.Type = None
    _buffer_type = None

    def __post_init__(self):
        if self.dynamic_fields is None and self.dataset_file is not None:
            metadata = read_yaml(self.dataset_file)
            self.dynamic_fields = {key: AIRRSequenceSet.STR_TO_TYPE[val]
                                   for key, val in metadata['type_dict_dynamic_fields'].items()}
        if self.bnp_dataclass is None and self.dynamic_fields is not None:
            self.bnp_dataclass = extend_dataclass_with_dynamic_fields(AIRRSequenceSet,
                                                                      list(self.dynamic_fields.items()))
        if self.identifier is None:
            self.identifier = uuid4().hex

    @classmethod
    def create_metadata_dict(cls, dataset_class, filename, type_dict, name, labels, identifier=None):
        return {"type_dict_dynamic_fields": {key: AIRRSequenceSet.TYPE_TO_STR[val] for key, val in type_dict.items()},
                "identifier": identifier if identifier is not None else uuid4().hex,
                "dataset_type": dataset_class if isinstance(dataset_class, str) else dataset_class.__name__,
                "filename": filename,
                "name": name,
                "labels": {} if labels is None else labels,
                "timestamp": str(datetime.now())}

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

    def get_label_names(self):
        invalid_label_names = ['type_dict_dynamic_fields', 'organism', 'j_gene', 'v_gene', 'j_call', 'v_call']
        return [el for el in list(self.labels.keys()) if el not in invalid_label_names] \
            if isinstance(self.labels, dict) else []


class SequenceDataset(ElementDataset):
    """A dataset class for sequence datasets (single chain). """

    @classmethod
    def build(cls, filename: Path, metadata_filename: Path, name: str = None, bnp_dc=None, labels: dict = None):
        metadata = read_yaml(metadata_filename)

        dynamic_fields = {key: AIRRSequenceSet.STR_TO_TYPE[val]
                          for key, val in metadata['type_dict_dynamic_fields'].items()}
        if bnp_dc is None:
            bnp_dc = extend_dataclass_with_dynamic_fields(AIRRSequenceSet, list(dynamic_fields.items()))

        if labels is None and 'labels' in metadata:
            labels = metadata['labels']

        return SequenceDataset(name=name, filename=filename, dataset_file=metadata_filename,
                               dynamic_fields=dynamic_fields, labels=labels,
                               bnp_dataclass=bnp_dc)

    @classmethod
    def build_from_objects(cls, sequences: List[ReceptorSequence], path: Path, name: str = None,
                           labels: dict = None, region_type: RegionType = RegionType.IMGT_CDR3):
        name = name if name is not None else uuid4().hex
        filename = path / f"{name}.tsv"

        all_fields_dict = make_all_fields_dict_from_sequences(sequences, region_type)
        bnp_dc, type_dict = build_dynamic_airr_sequence_set_dataclass(all_fields_dict)
        dc_object = bnp_dc(**all_fields_dict)
        bnp_write_to_file(filename, dc_object)

        dataset_metadata = cls.create_metadata_dict(dataset_class=cls.__name__,
                                                    filename=filename,
                                                    type_dict=type_dict,
                                                    name=name,
                                                    labels=labels)

        metadata_filename = path / f'{name}.yaml'
        write_dataset_yaml(metadata_filename, dataset_metadata)

        return SequenceDataset(filename=filename, name=name, labels=labels, dynamic_fields=type_dict,
                               dataset_file=metadata_filename, bnp_dataclass=bnp_dc,
                               identifier=dataset_metadata["identifier"])

    @classmethod
    def build_from_partial_df(cls, df: pd.DataFrame, path: Path, name: str = None, labels: dict = None,
                              type_dict: dict = None):

        airr_df = make_full_airr_seq_set_df(df)
        name = name if name is not None else uuid4().hex
        filename = path / f"{name}.tsv"

        airr_df.to_csv(filename, sep='\t', index=False)

        dataset_yaml = SequenceDataset.create_metadata_dict(SequenceDataset, filename=filename, type_dict=type_dict,
                                                            name=name, labels=labels)

        write_dataset_yaml(path / f'{name}.yaml', dataset_yaml)

        return SequenceDataset.build(filename, path / f'{name}.yaml', name)

    def get_metadata(self, field_names: list, return_df: bool = False):
        """Returns a dict or an equivalent pandas DataFrame with metadata information from ReceptorSequence objects for
        provided field names"""
        result = self.data.topandas()[field_names]
        result = fix_empty_strings_in_metadata(result)

        return result if return_df else result.to_dict("list")

    def get_attribute(self, attribute_name):
        return getattr(self.data, attribute_name)

    def make_subset(self, example_indices, path, dataset_type: str):
        data = self.data[example_indices]
        name = f"subset_{self.name}_{dataset_type}"
        data_filename = path / f'{name}.tsv'

        bnp_write_to_file(data_filename, data)

        metadata_filename = path / f'{name}.yaml'
        shutil.copyfile(self.dataset_file, metadata_filename)

        return SequenceDataset(filename=data_filename, name=name, labels=copy.deepcopy(self.labels),
                               dynamic_fields=self.dynamic_fields, dataset_file=metadata_filename,
                               bnp_dataclass=self.bnp_dataclass)

    def get_example_count(self):
        if self.element_count is None:
            self.element_count = len(self.data)
        return self.element_count

    def get_data(self, batch_size: int = 1, region_type: RegionType = RegionType.IMGT_CDR3):
        return make_sequences_from_data(self.data, self.dynamic_fields, region_type)

    def get_example_ids(self):
        return self.data.sequence_id.tolist()

    def get_data_from_index_range(self, start_index: int, end_index: int):
        return self.data[start_index: end_index]


class ReceptorDataset(ElementDataset):
    """A dataset class for receptor datasets (paired chain)."""

    @classmethod
    def build(cls, filename: Path, metadata_filename: Path, name: str = None, bnp_dc=None, labels: dict = None):
        metadata = read_yaml(metadata_filename)
        dynamic_fields = {k: AIRRSequenceSet.STR_TO_TYPE[v] for k, v in metadata['type_dict_dynamic_fields'].items()}

        if bnp_dc is None:
            bnp_dc = extend_dataclass_with_dynamic_fields(AIRRSequenceSet, list(dynamic_fields.items()))

        if labels is None and 'labels' in metadata:
            labels = metadata['labels']

        return ReceptorDataset(name=name, filename=filename, dataset_file=metadata_filename,
                               dynamic_fields=dynamic_fields, labels=labels,
                               bnp_dataclass=bnp_dc)

    @classmethod
    def build_from_objects(cls, receptors: List[Receptor], path: Path, name: str = None,
                           labels: dict = None, region_type: RegionType = RegionType.IMGT_CDR3):
        name = name if name is not None else uuid4().hex
        filename = path / f"{name}.tsv"

        all_fields_dict = make_all_fields_dict_from_receptors(receptors, region_type)
        bnp_dc, type_dict = build_dynamic_airr_sequence_set_dataclass(all_fields_dict)
        bnp_write_to_file(filename, bnp_dc(**all_fields_dict))

        metadata_filename = path / f'{name}.yaml'
        dataset_metadata = cls.create_metadata_dict(dataset_class=cls.__name__,
                                                    filename=filename,
                                                    type_dict=type_dict,
                                                    name=name,
                                                    labels=labels)

        write_dataset_yaml(metadata_filename, dataset_metadata)

        return ReceptorDataset(filename=filename, name=name, labels=labels, dynamic_fields=type_dict,
                               dataset_file=metadata_filename, bnp_dataclass=bnp_dc,
                               identifier=dataset_metadata['identifier'])

    def get_metadata(self, field_names: list, return_df: bool = False):
        """Returns a dict or an equivalent pandas DataFrame with metadata information from Receptor objects for
        provided field names"""
        result = fix_empty_strings_in_metadata(self.data.topandas())

        result = result.groupby('cell_id', sort=False).agg(
            lambda x: "_".join([str(el) for el in list(set(x))])).reset_index()
        if any(field_name not in result.columns for field_name in field_names):
            logging.warning(
                f"ReceptorDataset {self.name}: not all requested metadata fields are available: {field_names}.")
        else:
            result = result[field_names]

        return result if return_df else result.to_dict('list')

    def make_subset(self, example_indices, path, dataset_type: str):
        true_indices = np.array([[ind * 2, ind * 2 + 1] for ind in example_indices]).flatten()
        data = self.data[true_indices]
        name = f"{dataset_type}_subset_{self.name}"

        bnp_write_to_file(path / f"{name}.tsv", data)

        metadata_filename = path / f'{name}.yaml'
        metadata = read_yaml(self.dataset_file)
        write_dataset_yaml(metadata_filename, {
            **metadata, **{'filename': f"{name}.tsv", 'name': name}
        })

        return ReceptorDataset(filename=path / f"{name}.tsv", name=name,
                               labels=copy.deepcopy(self.labels),
                               dynamic_fields=self.dynamic_fields, dataset_file=metadata_filename,
                               bnp_dataclass=self.bnp_dataclass)

    def get_example_count(self):
        if self.element_count is None:
            self.element_count = len(self.data) // 2
        return self.element_count

    def get_data(self, batch_size: int = 1, region_type: RegionType = RegionType.IMGT_CDR3):
        return make_receptors_from_data(self.data, self.dynamic_fields,
                                        f"ReceptorDataset {self.identifier}", region_type)

    def get_example_ids(self):
        return np.unique(self.data.cell_id.tolist()).tolist()

    def get_data_from_index_range(self, start_index: int, end_index: int):
        return self.data[start_index * 2: end_index * 2]


def make_all_fields_dict_from_receptors(receptors: List[Receptor], region_type: RegionType = RegionType.IMGT_CDR3):
    sequences = list(itertools.chain.from_iterable([r.chain_1, r.chain_2] for r in receptors))
    for index, receptor in enumerate(receptors):
        sequences[index * 2].metadata = {**sequences[index * 2].metadata, **receptor.metadata,
                                         'receptor_id': receptor.receptor_id, 'cell_id': receptor.cell_id}
        sequences[index * 2 + 1].metadata = {**sequences[index * 2 + 1].metadata, **receptor.metadata,
                                             'receptor_id': receptor.receptor_id, 'cell_id': receptor.cell_id}

    return make_all_fields_dict_from_sequences(sequences, region_type)


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

        for missing_key in dynamic_fields.keys():
            if len(dynamic_fields[missing_key]) == index:
                dynamic_fields[missing_key].append('')

        keys_not_to_add = ['metadata', 'sequence_aa', 'sequence'] + list(sequence.metadata.keys())
        for key in [f.name for f in fields(ReceptorSequence) if f.name not in keys_not_to_add]:
            all_fields[key].append(getattr(sequence, key))

        all_fields[region_type.value].append(sequence.sequence)
        all_fields[region_type.value + "_aa"].append(sequence.sequence_aa)

    all_fields = fill_in_neutral_vals(all_fields, airr_fields, sequences)

    return {**all_fields, **dynamic_fields}


def fill_in_neutral_vals(all_fields, airr_fields, sequences):
    for f in airr_fields:
        neutral_val = AIRRSequenceSet.get_neutral_value(f.type)
        if len(all_fields[f.name]) == 0:
            all_fields[f.name] = [neutral_val for _ in range(len(sequences))]
        elif any(val is None for val in all_fields[f.name]):
            all_fields[f.name] = [val if val is not None else neutral_val for val in all_fields[f.name]]

    return all_fields


def fix_empty_strings_in_metadata(df: pd.DataFrame):
    for col, col_type in df.dtypes.to_dict().items():
        if col_type == object:
            df[col] = df[col].astype(str).replace('nan', '')
    return df
