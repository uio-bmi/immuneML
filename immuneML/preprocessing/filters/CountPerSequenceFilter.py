import copy
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.preprocessing.filters.Filter import Filter


class CountPerSequenceFilter(Filter):
    """
    Removes all sequences from a Repertoire when they have a count below low_count_limit, or sequences with no count
    value if remove_without_counts is True.
    This filter can be applied to Repertoires and RepertoireDatasets.

    Arguments:

        low_count_limit (int): The inclusive minimal count value in order to retain a given sequence.

        remove_without_count (bool): Whether the sequences without a reported count value should be removed.

        remove_empty_repertoires (bool): Whether repertoires without sequences should be removed.
        Only has an effect when remove_without_count is also set to True.

        batch_size (int): number of repertoires that can be loaded at the same time (only affects the speed when applying this filter on a RepertoireDataset)


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    CountPerSequenceFilter:
                        remove_without_count: True
                        remove_empty_repertoires: True
                        low_count_limit: 3
                        batch_size: 4

    """

    def __init__(self, low_count_limit: int, remove_without_count: bool, remove_empty_repertoires: bool,  batch_size: int):
        self.low_count_limit = low_count_limit
        self.remove_without_count = remove_without_count
        self.remove_empty_repertoires = remove_empty_repertoires
        self.batch_size = batch_size

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        processed_dataset = copy.deepcopy(dataset)

        with Pool(params["batch_size"]) as pool:
            repertoires = pool.starmap(CountPerSequenceFilter.process_repertoire,
                                       [(repertoire, params) for repertoire in dataset.repertoires])

        if params["remove_empty_repertoires"]:
            repertoires = Filter.remove_empty_repertoires(repertoires)

        processed_dataset.repertoires = repertoires

        Filter.check_dataset_not_empty(processed_dataset, "CountPerSequenceFilter")

        return processed_dataset

    @staticmethod
    def process_repertoire(repertoire: Repertoire, params: dict) -> Repertoire:

        counts = repertoire.get_counts()
        counts = counts if counts is not None else np.full(repertoire.get_element_count(), None)
        not_none_indices = counts != None
        counts[not_none_indices] = counts[not_none_indices].astype(np.int)
        indices_to_keep = np.full(repertoire.get_element_count(), False)
        if params["remove_without_count"] and params["low_count_limit"] is not None:
            np.greater_equal(counts, params["low_count_limit"], out=indices_to_keep, where=not_none_indices)
        elif params["remove_without_count"]:
            indices_to_keep = not_none_indices
        elif params["low_count_limit"] is not None:
            indices_to_keep[np.logical_not(not_none_indices)] = True
            np.greater_equal(counts, params["low_count_limit"], out=indices_to_keep, where=not_none_indices)

        processed_repertoire = Repertoire.build_like(repertoire, indices_to_keep, params["result_path"], filename_base=f"{repertoire.data_filename.stem}_filtered")

        return processed_repertoire

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path) -> RepertoireDataset:
        params = {"result_path": result_path, "low_count_limit": self.low_count_limit, "remove_without_count": self.remove_without_count,
                  "remove_empty_repertoires": self.remove_empty_repertoires, "batch_size": self.batch_size}
        return CountPerSequenceFilter.process(dataset, params)
