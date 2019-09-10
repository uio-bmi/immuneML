import copy
import os

import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.preprocessing.Preprocessor import Preprocessor
from source.util.PathBuilder import PathBuilder


class DatasetChainFilter(Preprocessor):
    """
    Preprocessing filter which removes all repertoires from the RepertoireDataset object which contain at least one sequence
    from chain different than "keep_chain" parameter
    """

    def __init__(self, keep_chain: Chain, result_path: str = None):
        self.keep_chain = keep_chain
        self.result_path = result_path

    def process_dataset(self, dataset: RepertoireDataset):
        return DatasetChainFilter.process(dataset=dataset, params={"keep_chain": self.keep_chain, "result_path": self.result_path})

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        processed_dataset = copy.deepcopy(dataset)
        PathBuilder.build(params["result_path"])
        filenames = []
        indices = []
        for index, repertoire in enumerate(dataset.get_data()):
            if all(sequence.metadata.chain == Chain[params["keep_chain"].upper()] for sequence in repertoire.sequences):
                filename = params["result_path"] + "{}.pickle".format(repertoire.identifier)
                os.rename(dataset.get_filenames()[index], filename)
                filenames.append(filename)
                indices.append(index)

        processed_dataset.metadata_file = DatasetChainFilter.build_new_metadata(processed_dataset, indices, params["result_path"])
        processed_dataset.set_filenames(filenames)
        return processed_dataset

    @staticmethod
    def build_new_metadata(dataset: RepertoireDataset, indices_to_keep: list, result_path: str):
        if dataset.metadata_file:
            df = pd.read_csv(dataset.metadata_file, index_col=0).iloc[indices_to_keep, :]
            for index, row in df.iterrows():
                row["filename"] = dataset.get_filenames()[index]
            path = result_path + "/{}_dataset_chain_filtered.csv" \
                .format(os.path.splitext(os.path.basename(dataset.metadata_file))[0])
            df.to_csv(path)
        else:
            path = None
        return path
