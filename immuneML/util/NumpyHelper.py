import json
from enum import Enum

import numpy as np


class NumpyHelper:

    @staticmethod
    def group_structured_array_by(data, field):
        for col in data.dtype.names:
            data[col][np.argwhere(data[col] == None)] = ""
        sorted_data = np.sort(data, order=[field], axis=0)
        grouped_lists = np.split(sorted_data, np.cumsum(np.unique(sorted_data[field], return_counts=True)[1])[:-1])
        return grouped_lists

    @staticmethod
    def is_simple_type(t):
        """returns if the type t is string or a number so that it does not use pickle if serialized"""
        return t in [str, int, float, np.str_, np.int_, np.float_]

    @staticmethod
    def get_numpy_representation(obj):
        """converts object to representation that can be stored without pickle enables in numpy arrays; if it is an object or a dict,
        it will be serialized to a json string"""

        if obj is None:
            return '{}'
        elif NumpyHelper.is_simple_type(type(obj)):
            return obj
        elif not isinstance(obj, dict):
            try:
                representation = vars(obj)
            except Exception as e:
                print(e)
                print(obj)
                print(type(obj))
                raise e
        else:
            representation = obj

        return json.dumps(representation, default=lambda x: NumpyHelper.convert_to_str(x))

    @staticmethod
    def convert_to_str(x):
        if isinstance(x, Enum):
            return x.name
        else:
            return str(x)
