import numpy as np


class NumpyHelper:

    @staticmethod
    def group_structured_array_by(data, field):
        for col in data.dtype.names:
            data[col][np.argwhere(data[col] == None)] = ""
        sorted_data = np.sort(data, order=[field], axis=0)
        grouped_lists = np.split(sorted_data, np.cumsum(np.unique(sorted_data[field], return_counts=True)[1])[:-1])
        return grouped_lists
