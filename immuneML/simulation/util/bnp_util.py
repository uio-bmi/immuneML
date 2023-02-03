from dataclasses import fields as get_fields
from dataclasses import make_dataclass as dc_make_dataclass
from itertools import chain
from typing import List

import numpy as np
from bionumpy import as_encoded_array, EncodedArray
from bionumpy.bnpdataclass import BNPDataClass, bnpdataclass
from bionumpy.encodings import Encoding


def make_bnp_dataclass_object_from_dicts(dict_objects: List[dict], field_type_map: dict = None, signals: list = None, base_class=None) -> BNPDataClass:
    if not isinstance(dict_objects, list) or len(dict_objects) == 0:
        raise RuntimeError("Cannot make dataclass, got empty list as input.")

    transformed_objs = _list_of_dicts_to_dict_of_lists(dict_objects)
    fields = _extract_fields(transformed_objs, field_type_map)
    fields_list = sorted(list(fields.items()), key=lambda x: x[0])

    if signals is not None:
        signal_names = [field for field in fields if field in [signal.id for signal in signals]]
        functions = {"get_signal_matrix": lambda self: np.array([getattr(self, name) for name in signal_names]).T,
                     "get_signal_names": lambda self: signal_names}

        new_class = bnpdataclass(dc_make_dataclass("DynamicDC", bases=tuple([base_class]) if base_class is not None else (), namespace=functions,
                                                   fields=fields_list))
    else:
        new_class = base_class.extend(fields)

    for key in transformed_objs:
        if transformed_objs[key] is None and isinstance(fields[key], Encoding):
            transformed_objs[key] = ['' for _ in range(len(dict_objects))]

    return new_class(**transformed_objs)


def _extract_fields(transformed_objs, field_type_map):
    fields = {}
    for field_name in sorted(list(transformed_objs.keys())):
        assert all(isinstance(val, type(transformed_objs[field_name][0])) for val in transformed_objs[field_name]), \
            [type(val) for val in transformed_objs[field_name]]

        if field_type_map is not None and field_name in field_type_map:
            field_type = field_type_map[field_name]
            if isinstance(field_type, Encoding):
                transformed_objs[field_name] = as_encoded_array(transformed_objs[field_name], field_type) if any(transformed_objs[field_name]) else None
        elif isinstance(transformed_objs[field_name][0], EncodedArray):
            field_type = transformed_objs[field_name][0].encoding
        elif transformed_objs[field_name] is not None:
            field_type = type(transformed_objs[field_name][0])
        else:
            field_type = str

        fields[field_name] = field_type

    return fields


def merge_dataclass_objects(objects: list):  # TODO: replace with equivalent from npstructures
    field_names = sorted(list(set(chain.from_iterable([field.name for field in get_fields(obj)] for obj in objects))))

    for obj in objects:
        assert all(hasattr(obj, field) for field in field_names), (obj, field_names)

    cls = type(objects[0])
    return cls(**{field_name: list(chain.from_iterable([getattr(obj, field_name) for obj in objects])) for field_name in field_names})


def _make_new_fields(new_fields: dict) -> List[tuple]:
    fields = []

    for field_name, field_vals in new_fields.items():
        assert all(isinstance(field_val, type(field_vals[0])) for field_val in field_vals)
        fields.append((field_name, type(field_vals[0])))

    fields.sort(key=lambda x: x[0])

    return fields


def _list_of_dicts_to_dict_of_lists(dict_objects: List[dict]):
    field_names = list(set(chain.from_iterable(list(obj.keys()) for obj in dict_objects)))

    transformed_objs = {field_name: [] for field_name in field_names}

    for obj in dict_objects:
        assert all(key in field_names for key in obj.keys()), (list(obj.keys()), field_names)
        for field_name in field_names:
            transformed_objs[field_name].append(obj[field_name])

    return transformed_objs
