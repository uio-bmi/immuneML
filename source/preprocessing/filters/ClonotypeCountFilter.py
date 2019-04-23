import copy

from source.data_model.dataset.Dataset import Dataset
from source.preprocessing.Preprocessor import Preprocessor


class ClonotypeCountFilter(Preprocessor):

    @staticmethod
    def process(dataset: Dataset, params: dict) -> Dataset:
        processed_dataset = copy.deepcopy(dataset)
        filenames = []
        for index, repertoire in enumerate(dataset.get_data()):
            if "lower_limit" in params.keys() and len(repertoire.sequences) >= params["lower_limit"] or \
                "upper_limit" in params.keys() and len(repertoire.sequences) <= params["upper_limit"]:
                filenames.append(dataset.filenames[index])
        processed_dataset.filenames = filenames
        return processed_dataset