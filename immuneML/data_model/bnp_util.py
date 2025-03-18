import dataclasses
import logging
from dataclasses import fields as get_fields
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, Any, Tuple, List

import bionumpy as bnp
import numpy as np
import pandas as pd
import yaml
from bionumpy.bnpdataclass import bnpdataclass
from bionumpy.encoded_array import EncodedArray

from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import Chain, RegionType
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.ReflectionHandler import ReflectionHandler


def bnp_write_to_file(filename: Path, bnp_object):
    buffer_type = bnp.io.delimited_buffers.get_bufferclass_for_datatype(type(bnp_object), delimiter='\t',
                                                                        has_header=True)
    with bnp.open(str(filename), 'w', buffer_type=buffer_type) as file:
        file.write(bnp_object)


def bnp_read_from_file(filename: Path, buffer_type: bnp.io.delimited_buffers.DelimitedBuffer = None, dataclass=None):
    if buffer_type is None:
        buffer_type = bnp.io.delimited_buffers.get_bufferclass_for_datatype(dataclass, delimiter='\t', has_header=True)
    with bnp.open(str(filename), buffer_type=buffer_type) as file:
        return file.read()  # TODO: fix - throws error when empty file (no lines after header)


def write_dataset_yaml(filename: Path, yaml_dict):
    for mandatory_field in ["identifier", "dataset_type", "name", "labels"]:
        assert mandatory_field in yaml_dict.keys(), f"Error exporting {filename.stem}: missing field {mandatory_field}"

    if yaml_dict["dataset_type"] == "RepertoireDataset":
        assert "metadata_file" in yaml_dict.keys(), f"Error exporting {filename.stem}: missing field metadata_file"

    if yaml_dict["dataset_type"] in ("SequenceDataset", "ReceptorDataset"):
        assert "filename" in yaml_dict.keys(), f"Error exporting {filename.stem}: missing field filename"
        assert "type_dict_dynamic_fields" in yaml_dict.keys(), f"Error exporting {filename.stem}: missing field type_dict_dynamic_fields"

    assert type(yaml_dict["labels"]) == dict or type(yaml_dict["labels"]) == None, "labels format must be dict or None"

    write_yaml(filename, yaml_dict)


def write_yaml(filename: Path, yaml_dict):
    with filename.open('w') as file:
        yaml.dump({k: str(v) if isinstance(v, Path) else v for k, v in yaml_dict.items()}, file)
    return filename


def read_yaml(filename: Path) -> dict:
    with filename.open("r") as file:
        content = yaml.safe_load(file)
    return load_type_dict(content)


def get_sequence_field_name(region_type: RegionType = RegionType.IMGT_CDR3,
                            sequence_type: SequenceType = SequenceType.AMINO_ACID):
    suffix = "_aa" if sequence_type == SequenceType.AMINO_ACID else ""
    return region_type.value + suffix


def load_type_dict(full_dict: dict) -> dict:
    if 'type_dict' in full_dict:
        full_dict['type_dict'] = {key: AIRRSequenceSet.STR_TO_TYPE[val] for key, val in full_dict['type_dict'].items()}
    return full_dict


def build_dynamic_bnp_dataclass(all_fields_dict: Dict[str, Any]):
    sequence_field_names = {field.name: field.type for field in dataclasses.fields(AIRRSequenceSet)}
    types = {}

    for key, value in all_fields_dict.items():
        if key in sequence_field_names:
            field_type = sequence_field_names[key]
        else:
            field_type = get_field_type_from_values(value)
        types[key] = field_type

    dc = AIRRSequenceSet.extend(tuple((name, t, AIRRSequenceSet.get_neutral_value(t)) for name, t in types.items()
                                      if name not in list(AIRRSequenceSet.get_field_type_dict().keys())))
    return dc, types


def build_dynamic_bnp_dataclass_obj(all_fields_dict: Dict[str, Any]):
    dc, types = build_dynamic_bnp_dataclass(all_fields_dict)
    all_fields_dict = add_neutral_values(all_fields_dict, types)
    all_fields_dict = convert_to_expected_types(all_fields_dict, types)
    all_fields_dict = make_full_airr_seq_set_df(pd.DataFrame(all_fields_dict)).to_dict(orient='list')
    return dc(**all_fields_dict), types


def convert_to_expected_types(all_fields_dict, types) -> dict:
    for field_name, field_type in types.items():
        all_fields_dict[field_name] = [field_type(val) for val in all_fields_dict[field_name]]

    return all_fields_dict


def prepare_values_for_bnp(field_values: dict, types: dict) -> dict:
    values = add_neutral_values(field_values, types)
    values = convert_enums_to_str(values)
    return values


def add_neutral_values(field_values: dict, types: dict) -> dict:
    return {
        field: [val if val is not None else AIRRSequenceSet.get_neutral_value(types[field]) for val in values]
        for field, values in field_values.items()
    }


def convert_enums_to_str(field_values: dict) -> dict:
    return {
        field: [val.to_string() if isinstance(val, Enum) and hasattr(val, 'to_string') else val for val in values]
        for field, values in field_values.items()
    }


def extend_dataclass_with_dynamic_fields(cls, fields: List[Tuple[str, type]], cls_name: str = None):
    cls_name = cls_name if cls_name is not None else "Dynamic" + cls.__name__
    new_cls = bnpdataclass(dataclasses.make_dataclass(cls_name,
                                                      fields=[(name, field_type, dataclasses.field(default=None)) for
                                                              name, field_type in fields], bases=(cls,)))

    def dynamic_fields(cls):
        return [el[0] for el in fields]

    new_cls.dynamic_fields = classmethod(dynamic_fields)

    return new_cls


def get_field_type_from_values(values):
    if isinstance(values, np.ndarray):
        t = type(values[0].item())
    elif len(values) == 0:
        t = str
    elif values[0] is not None:
        if issubclass(type(values[0]), Enum):
            t = str
        else:
            t = type(values[0])
    else:
        proper_values = [v for v in values if v is not None]
        if len(proper_values) > 0:
            t = type(proper_values[0])
        else:
            t = str

    return t


def get_row_by_index(self, index) -> dict:
    field_names = [f.name for f in dataclasses.fields(self)]
    d = dict()

    for field_name in field_names:
        field_value = getattr(self, field_name)[index]

        if isinstance(field_value, EncodedArray):
            field_value = field_value.to_string()
        else:
            field_value = field_value.item()

        d[field_name] = field_value

    return d


def get_rows_by_indices(self, index1, index2) -> dict:
    row1 = self.get_row_by_index(index1)
    row2 = self.get_row_by_index(index2)

    assert row1['cell_id'] == row2['cell_id'], (row1['cell_id'], row2['cell_id'])

    return {
        **{f'{Chain.get_chain(row1["locus"]).name.lower()}_{key}': val for key, val in row1.items()},
        **{f'{Chain.get_chain(row2["locus"]).name.lower()}_{key}': val for key, val in row2.items()},
    }


def get_single_row_value(self, attr_name: str):
    if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
        val = getattr(self, attr_name)
        if isinstance(val, EncodedArray):
            return val.to_string()
        elif isinstance(val, np.ndarray):
            return val.item()
    else:
        return None


def make_element_dataset_objects(bnp_object, class_name) -> list:
    cls = ReflectionHandler.get_class_by_name(class_name, 'data_model')
    if class_name == 'ReceptorSequence':
        objects = [cls.create_from_record(**bnp_object.get_row_by_index(i)) for i in range(len(bnp_object))]
    else:
        objects = [cls.create_from_record(**bnp_object.get_rows_by_indices(i, i + 1)) for i in
                   range(0, len(bnp_object), 2)]
    return objects


def make_buffer_type_from_dataset_file(dataset_file: Path):
    if dataset_file is not None and dataset_file.exists() and dataset_file.is_file():
        with dataset_file.open('r') as file:
            metadata = yaml.safe_load(file)

        type_dict = {key: AIRRSequenceSet.STR_TO_TYPE[val] for key, val in metadata["type_dict"].items()}
        dataclass = extend_dataclass_with_dynamic_fields(AIRRSequenceSet, list(type_dict.items()))
        return bnp.io.delimited_buffers.get_bufferclass_for_datatype(dataclass, delimiter='\t', has_header=True)
    else:
        raise RuntimeError(f"Dataset file {dataset_file} doesn't exist, cannot load the dataset.")


def merge_dataclass_objects(objects: list, fill_unmatched: bool = False):
    fields = {k: v for d in [{field.name: field.type for field in get_fields(obj)} for obj in objects] for k, v in
              d.items()}
    fields = {k: fields[k] for k in sorted(list(fields.keys()))}

    tmp_objs = []

    for obj in objects:
        missing_fields = [field for field in fields.keys() if not hasattr(obj, field)]

        if not fill_unmatched or len(missing_fields) == 0:
            assert all(hasattr(obj, field) for field in fields.keys()), (obj, list(fields.keys()))
            tmp_objs.append(obj)
        else:
            logging.info(f"Filling in missing fields when merging dataclass objects: {missing_fields}\nObject:\n{obj}")
            tmp_objs.append(
                obj.add_fields({field_name: [AIRRSequenceSet.get_neutral_value(fields[field_name])] * len(obj)
                                for field_name in missing_fields},
                               {field_name: field_type for field_name, field_type in fields.items() if
                                field_name in missing_fields}))

    cls = type(tmp_objs[0])
    return cls(
        **{field_name: list(chain.from_iterable([getattr(obj, field_name) for obj in tmp_objs])) for field_name in
           fields.keys()})


def get_type_dict_from_bnp_object(bnp_object) -> dict:
    return {field.name: field.type for field in get_fields(bnp_object)}


def make_full_airr_seq_set_df(df):
    field_type_dict = AIRRSequenceSet.get_field_type_dict()
    default_fields = pd.DataFrame({
        f_name: [AIRRSequenceSet.get_neutral_value(f_type) for _ in range(df.shape[0])]
        for f_name, f_type in field_type_dict.items() if f_name not in df.columns})
    return pd.concat([df, default_fields], axis=1)
