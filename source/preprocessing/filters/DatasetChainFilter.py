import copy
import os

import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.preprocessing.Preprocessor import Preprocessor


class DatasetChainFilter(Preprocessor):

    def __init__(self, keep_chain: Chain):
        self.keep_chain = keep_chain

    def process_dataset(self, dataset: RepertoireDataset):
        return DatasetChainFilter.process(dataset=dataset, params={"keep_chain": self.keep_chain})

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        processed_dataset = copy.deepcopy(dataset)
        filenames = []
        indices = []
        for index, repertoire in enumerate(dataset.get_data()):
            if all(sequence.metadata.chain == Chain[params["keep_chain"].upper()] for sequence in repertoire.sequences):
                filename = dataset.get_filenames()[index].replace(os.path.basename(dataset.get_filenames()[index]),
                                                                  "{}.pickle".format(repertoire.identifier))
                os.rename(dataset.get_filenames()[index], filename)
                filenames.append(filename)
                indices.append(index)

        processed_dataset.metadata_file = DatasetChainFilter.build_new_metadata(processed_dataset, indices)
        processed_dataset.set_filenames(filenames)
        return processed_dataset

    @staticmethod
    def build_new_metadata(dataset: RepertoireDataset, indices_to_keep: list):
        if dataset.metadata_file:
            df = pd.read_csv(dataset.metadata_file, index_col=0).iloc[indices_to_keep, :]
            path = os.path.dirname(os.path.abspath(dataset.metadata_file)) + "/{}_dataset_chain_filtered.csv" \
                .format(os.path.splitext(os.path.basename(dataset.metadata_file))[0])
            df.to_csv(path)
        else:
            path = None
        return path