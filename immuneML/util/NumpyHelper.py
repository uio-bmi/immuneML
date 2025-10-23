import json
import logging
from enum import Enum

import numpy as np

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


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
    def create_memmap_array_in_cache(shape: tuple, data: np.ndarray = None) -> np.ndarray:
        """Creates a memory-mapped array and optionally initializes it with data."""
        import uuid
        dir_path = PathBuilder.build(EnvironmentSettings.get_cache_path() / "memmap_storage")
        memmap_path = dir_path / f"temp_{uuid.uuid4()}.npy"
        if data is not None:
            data.astype('float32').tofile(memmap_path)
            return np.memmap(memmap_path, dtype='float32', mode='r+', shape=shape)
        else:
            return np.memmap(memmap_path, dtype='float32', mode='w+', shape=shape)

    @staticmethod
    def concat_arrays_rowwise(arrays: list, force='auto', dense_max_mb=100, use_memmap=False):
        """
        Concatenate 2D numpy arrays or sparse matrices row-wise.

        Parameters
        ----------
        arrays : list of np.ndarray or scipy.sparse matrices
        force : {"auto", "dense", "sparse"}
            - "auto": use memory-based heuristic (default)
            - "dense": always return numpy.ndarray
            - "sparse": always return scipy.sparse.csr_matrix
        dense_max_mb : int
            Threshold for converting sparse -> dense in "auto" mode.
        use_memmap: bool
        """
        if not arrays:
            raise ValueError("No matrices provided")

        from scipy import sparse
        if any(sparse.issparse(array) for array in arrays):
            # Convert all to sparse (CSR for efficiency)
            matrices = [array.astype(np.float32) if sparse.issparse(array)
                        else sparse.csr_matrix(array).astype(np.float32) for array in arrays]
            result = sparse.hstack(matrices, format="csr")

            # Estimate dense memory size
            size_in_mb = result.shape[0] * result.shape[1] * result.dtype.itemsize / (1024 * 1024)
            if size_in_mb <= dense_max_mb:
                result = result.toarray()
        else:
            # All are numpy arrays
            result = np.hstack(arrays)

        if np.isnan(result).any():
            import inspect
            logging.error(f"NumpyHelper: NaN values found in concatenated array; called from {inspect.stack()[1].function}")
            raise RuntimeError('NumpyHelper: NaN values found in concatenated array')

        if force == "dense" and sparse.issparse(result):
            result = result.toarray()
        elif force == "sparse" and not sparse.issparse(result):
            result = sparse.csr_matrix(result)

        if use_memmap:
            result = NumpyHelper.create_memmap_array_in_cache(result.shape, result)

        return result

