# quality: gold

import pickle

from source.IO.dataset_export.DataExporter import DataExporter
from source.data_model.dataset.Dataset import Dataset
from source.util.PathBuilder import PathBuilder


class PickleExporter(DataExporter):

    @staticmethod
    def export(dataset: Dataset, path, filename):
        PathBuilder.build(path)
        with open(path + filename, "wb") as file:
            pickle.dump(dataset, file, pickle.HIGHEST_PROTOCOL)
