from dataclasses import make_dataclass
from itertools import chain
from typing import List

from bionumpy import DNAEncoding, AminoAcidEncoding, as_encoded_array
from bionumpy.bnpdataclass import bnpdataclass


def make_bnp_dataclass_from_dicts(dict_objects: List[dict]):
    if not isinstance(dict_objects, list) or len(dict_objects) == 0:
        raise RuntimeError("Cannot make dataclass, got empty list as input.")

    transformed_objs = _list_of_dicts_to_dict_of_lists(dict_objects)

    fields = []
    for field_name in transformed_objs.keys():
        assert all(isinstance(val, type(transformed_objs[field_name][0])) for val in transformed_objs[field_name]), \
            [type(val) for val in transformed_objs[field_name]]

        if field_name in ['sequence', 'sequence_aa']:
            field_type = DNAEncoding if field_name == "sequence" else AminoAcidEncoding
            transformed_objs[field_name] = as_encoded_array(transformed_objs[field_name], field_type)
        else:
            field_type = type(transformed_objs[field_name][0])

        fields.append((field_name, field_type))

    new_class = bnpdataclass(make_dataclass('DynamicDC', fields=fields))
    return new_class(**transformed_objs)


def merge_dataclass_objects(objects: List[bnpdataclass]):
    field_names = list(set(chain.from_iterable(list(obj.__annotations__.keys()) for obj in objects)))

    for obj in objects:
        assert all(hasattr(obj, field) for field in field_names), (obj, field_names)

    cls = type(objects[0])
    return cls(**{key: chain.from_iterable([getattr(obj, key) for obj in objects]) for key in objects[0].__annotations__.keys()})


def add_field_to_bnp_dataclass(original_object, new_field_name, new_field_type, new_field_value):
    assert all(isinstance(val, new_field_type) for val in new_field_value), (new_field_name, new_field_type, new_field_value)
    return add_fields_to_bnp_dataclass(original_object, {new_field_name: new_field_value})


def add_fields_to_bnp_dataclass(original_object: bnpdataclass, new_fields: dict):
    original_class = type(original_object)
    base_fields = [(field_name, field_type) for field_name, field_type in original_class.__annotations__.items()]
    functions = {func: getattr(original_class, func) for func in dir(original_class)
                 if callable(getattr(original_class, func)) and not func.startswith("__")}

    new_cls = bnpdataclass(make_dataclass('DynamicDC',
                                          fields=base_fields + _make_new_fields(new_fields),
                                          namespace=functions))

    return new_cls(**{**vars(original_object), **new_fields})


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
