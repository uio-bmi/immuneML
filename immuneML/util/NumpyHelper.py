import json
from enum import Enum

import numpy as np

from immuneML.caching.CacheHandler import CacheHandler

class NumpyHelper:

    SIMPLE_TYPES = [str, int, float, bool, np.str_, np.int_, np.float_, np.bool_]

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
        return t in NumpyHelper.SIMPLE_TYPES

    @staticmethod
    def get_numpy_representation(obj):
        """converts object to representation that can be stored without pickle enables in numpy arrays; if it is an object or a dict,
        it will be serialized to a json string"""

        if obj is None:
            return ''
        elif type(obj) in NumpyHelper.SIMPLE_TYPES:
            return obj
        elif isinstance(obj, Enum):
            return obj.name
        elif not isinstance(obj, dict) and not isinstance(obj, list) and not isinstance(obj, Enum):
            representation = vars(obj)
        else:
            representation = obj

        return json.dumps(representation, default=lambda x: x.name if isinstance(x, Enum) else str(x))

    @staticmethod
    def is_nan_or_empty(value):
        return value == 'nan' or value is None or (not isinstance(value, str) and np.isnan(value)) or value == ''

    @staticmethod
    def get_numpy_sequence_representation(dataset):
        return CacheHandler.memo_by_params((("dataset_identifier", dataset.identifier),
                                            "np_sequence_representation",
                                            ("example_ids", tuple(dataset.get_example_ids()))),
                                           lambda: NumpyHelper.compute_numpy_sequence_representation(dataset))

    @staticmethod
    def compute_numpy_sequence_representation(dataset, location=None):
        '''Computes an efficient unicode representation for SequenceDatasets where all sequences have the same length'''

        location = NumpyHelper.__name__ if location is None else location

        n_sequences = dataset.get_example_count()
        all_sequences = [None] * n_sequences
        sequence_length = None

        for i, sequence in enumerate(dataset.get_data()):
            sequence_str = sequence.get_sequence()
            all_sequences[i] = sequence_str

            if sequence_length is None:
                sequence_length = len(sequence_str)
            else:
                assert len(sequence_str) == sequence_length, f"{location}: expected all " \
                                                             f"sequences to be of length {sequence_length}, found " \
                                                             f"{len(sequence_str)}: '{sequence_str}'."

        unicode = np.array(all_sequences, dtype=f"U{sequence_length}")
        return unicode.view('U1').reshape(n_sequences, -1)