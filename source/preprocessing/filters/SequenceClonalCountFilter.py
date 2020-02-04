import copy
from multiprocessing.pool import Pool

import numpy as np

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.preprocessing.filters.Filter import Filter


class SequenceClonalCountFilter(Filter):
    """
    Removes sequences with counts below low_count_limit or the ones without count if remove_without_counts is True
    """

    def __init__(self, low_count_limit: int, remove_without_count: bool, batch_size: int):
        self.low_count_limit = low_count_limit
        self.remove_without_count = remove_without_count
        self.batch_size = batch_size

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        processed_dataset = copy.deepcopy(dataset)

        with Pool(params["batch_size"]) as pool:
            repertoires = pool.starmap(SequenceClonalCountFilter.process_repertoire,
                                       [(repertoire, params) for repertoire in dataset.repertoires])

        processed_dataset.repertoires = repertoires
        return processed_dataset

    @staticmethod
    def process_repertoire(repertoire: SequenceRepertoire, params: dict) -> SequenceRepertoire:

        counts = repertoire.get_counts()
        indices_to_keep = np.full(counts.shape[0], False)
        if params["remove_without_count"] and params["low_count_limit"] is not None:
            np.greater_equal(counts, params["low_count_limit"], out=indices_to_keep, where=counts != np.full_like(counts, None, order="F"))
        elif params["remove_without_count"]:
            indices_to_keep = counts != np.full(counts.shape, None)
        elif params["low_count_limit"] is not None:
            indices_to_keep = counts >= params["low_count_limit"]
        else:
            indices_to_keep = []

        processed_repertoire = SequenceRepertoire.build_like(repertoire, indices_to_keep, params["result_path"])
        return processed_repertoire

    def process_dataset(self, dataset: RepertoireDataset, result_path: str) -> RepertoireDataset:
        params = {"result_path": result_path}
        params["low_count_limit"] = self.low_count_limit
        params["remove_without_count"] = self.remove_without_count
        params["batch_size"] = self.batch_size
        return SequenceClonalCountFilter.process(dataset, params)
