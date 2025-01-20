from dataclasses import fields as get_fields
from itertools import chain

import numpy as np
from npstructures import RaggedArray


def pad_ragged_array(new_array, target_shape, padded_value):
    """pad ragged array to match sequence lengths"""
    padded_array = RaggedArray([[padded_value for _ in range(target_shape[1][ind])] for ind in range(target_shape[0])])
    for row_ind in range(target_shape[0]):
        np.put(padded_array[row_ind], 0, new_array[row_ind])

    return padded_array


def merge_dataclass_objects(objects: list):  # TODO: replace with equivalent from npstructures
    field_names = sorted(list(set(chain.from_iterable([field.name for field in get_fields(obj)] for obj in objects))))

    for obj in objects:
        assert all(hasattr(obj, field) for field in field_names), (obj, field_names)

    cls = type(objects[0])
    return cls(**{field_name: list(chain.from_iterable([getattr(obj, field_name) for obj in objects])) for field_name in
                  field_names})
