import copy
import os

from source.data_model.dataset.Dataset import Dataset
from source.preprocessing.Preprocessor import Preprocessor


class DatasetChainFilter(Preprocessor):

    @staticmethod
    def process(dataset: Dataset, params: dict) -> Dataset:
        processed_dataset = copy.deepcopy(dataset)
        filenames = []
        for index, repertoire in enumerate(dataset.get_data()):
            if all(sequence.metadata.chain == params["keep_chain"] for sequence in repertoire.sequences):
                filename = dataset.get_filenames()[index].replace(os.path.basename(dataset.get_filenames()[index]),
                                                                  "{}.pickle".format(repertoire.identifier))
                os.rename(dataset.get_filenames()[index], filename)
                filenames.append(filename)
        processed_dataset.set_filenames(filenames)
        return processed_dataset
