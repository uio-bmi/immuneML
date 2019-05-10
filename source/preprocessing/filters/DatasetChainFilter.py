import copy

from source.data_model.dataset.Dataset import Dataset
from source.preprocessing.Preprocessor import Preprocessor


class DatasetChainFilter(Preprocessor):

    @staticmethod
    def process(dataset: Dataset, params: dict) -> Dataset:
        processed_dataset = copy.deepcopy(dataset)
        filenames = []
        for index, repertoire in enumerate(dataset.get_data()):
            if all(sequence.metadata.chain == params["keep_chain"] for sequence in repertoire.sequences):
                filenames.append(dataset.filenames[index])
        processed_dataset.filenames = filenames
        return processed_dataset
