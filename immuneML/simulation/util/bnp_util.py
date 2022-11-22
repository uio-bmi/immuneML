from dataclasses import make_dataclass
from itertools import chain
from typing import List

import numpy as np
from bionumpy import DNAEncoding, AminoAcidEncoding, as_encoded_array, EncodedArray
from bionumpy.bnpdataclass import bnpdataclass


def _encode_dict_of_lists(transformed_objs: dict, fields: dict) -> dict:
    encoded_objs = {}
    for index, key in enumerate(fields.keys()):
        print(key)
        print(transformed_objs[key])
        if transformed_objs[key] is not None:
            if all(isinstance(el, EncodedArray) for el in transformed_objs[key]) and isinstance(transformed_objs[key], list):
                encoded_objs[key] = np.concatenate(transformed_objs[key])
            elif fields[key] != int:
                encoded_objs[key] = as_encoded_array(transformed_objs[key], fields[key])
            else:
                encoded_objs[key] = np.array(transformed_objs[key])

    print(encoded_objs)
    return encoded_objs


def make_bnp_dataclass_object_from_dicts(dict_objects: List[dict], encoding_dict: dict):
    if not isinstance(dict_objects, list) or len(dict_objects) == 0:
        raise RuntimeError("Cannot make dataclass, got empty list as input.")

    transformed_objs = _list_of_dicts_to_dict_of_lists(dict_objects)

    fields = {}
    for field_name in transformed_objs.keys():
        assert all(isinstance(val, type(transformed_objs[field_name][0])) for val in transformed_objs[field_name]), \
            [type(val) for val in transformed_objs[field_name]]

        if field_name in ['sequence', 'sequence_aa']:
            field_type = DNAEncoding if field_name == "sequence" else AminoAcidEncoding
            transformed_objs[field_name] = as_encoded_array(transformed_objs[field_name], field_type) if any(transformed_objs[field_name]) else None
        elif isinstance(transformed_objs[field_name][0], EncodedArray):
            field_type = type(transformed_objs[field_name][0].encoding)
        else:
            cls = type(transformed_objs[field_name][0])
            field_type = encoding_dict[cls.__name__] if cls.__name__ in encoding_dict else cls

        if transformed_objs[field_name] is not None:
            fields[field_name] = field_type

    print(fields)

    transformed_objs = _encode_dict_of_lists(transformed_objs, fields)

    new_class = bnpdataclass(make_dataclass('DynamicDC', fields=fields.items()))
    return new_class(**transformed_objs)


def merge_dataclass_objects(objects: list):
    field_names = list(set(chain.from_iterable(list(obj.__annotations__.keys()) for obj in objects)))

    for obj in objects:
        assert all(hasattr(obj, field) for field in field_names), (obj, field_names)

    cls = type(objects[0])
    return cls(**{field_name: chain.from_iterable([getattr(obj, field_name) for obj in objects]) for field_name in field_names})


def add_field_to_bnp_dataclass(original_object, new_field_name, new_field_type, new_field_value):
    assert all(isinstance(val, new_field_type) for val in new_field_value), (new_field_name, new_field_type, new_field_value)
    return add_fields_to_bnp_dataclass(original_object, {new_field_name: new_field_value})


def add_fields_to_bnp_dataclass(original_object, new_fields: dict):
    new_cls = make_new_bnp_dataclass(fields=_make_new_fields(new_fields), original_class=type(original_object))

    return new_cls(**{**vars(original_object), **new_fields})


def make_new_bnp_dataclass(fields: list, original_class=None):
    if original_class is not None:
        base_fields = [(field_name, field_type) for field_name, field_type in original_class.__annotations__.items()]
        functions = {func: getattr(original_class, func) for func in dir(original_class)
                     if callable(getattr(original_class, func)) and not func.startswith("__")}
    else:
        base_fields, functions = [], {}

    new_cls = bnpdataclass(make_dataclass('DynamicDC', fields=base_fields + fields, namespace=functions))
    return new_cls


def _make_new_fields(new_fields: dict) -> List[tuple]:
    fields = []

    for field_name, field_vals in new_fields.items():
        assert all(isinstance(field_val, type(field_vals[0])) for field_val in field_vals)
        fields.append((field_name, type(field_vals[0])))

    return fields


def _list_of_dicts_to_dict_of_lists(dict_objects: List[dict]):
    field_names = list(set(chain.from_iterable(list(obj.keys()) for obj in dict_objects)))

    transformed_objs = {field_name: [] for field_name in field_names}

    for obj in dict_objects:
        assert all(key in field_names for key in obj.keys()), (list(obj.keys()), field_names)
        for field_name in field_names:
            transformed_objs[field_name].append(obj[field_name])

    return transformed_objs
