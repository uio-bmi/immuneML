from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from immuneML.util.Logger import log
from immuneML.util.PathBuilder import PathBuilder


class PairwiseRepertoireComparison:

    @log
    def __init__(self, matching_columns: list, item_columns: list, path: Path, sequence_batch_size: int):
        self.matching_columns = matching_columns
        self.item_columns = item_columns
        self.path = PathBuilder.build(path)
        self.sequence_batch_size = sequence_batch_size
        self.comparison_data = None
        self.comparison_fn = None

    @log
    def create_comparison_data(self, dataset: RepertoireDataset) -> ComparisonData:

        comparison_data = ComparisonData(dataset.get_repertoire_ids(), self.matching_columns, self.sequence_batch_size, self.path)
        comparison_data.process_dataset(dataset)

        return comparison_data

    def prepare_caching_params(self, dataset: RepertoireDataset):
        return (
            ("dataset_identifier", dataset.identifier),
            ("item_attributes", self.item_columns)
        )

    def compare(self, dataset: RepertoireDataset, comparison_fn, comparison_fn_name):
        return CacheHandler.memo_by_params((("dataset_identifier", dataset.identifier),
                                            "pairwise_comparison",
                                            ("comparison_fn", comparison_fn_name)),
                                           lambda: self.compare_repertoires(dataset, comparison_fn))

    def memo_by_params(self, dataset: RepertoireDataset):
        comparison_data = CacheHandler.memo_by_params(self.prepare_caching_params(dataset), lambda: self.create_comparison_data(dataset))
        return comparison_data

    @log
    def compare_repertoires(self, dataset: RepertoireDataset, comparison_fn):
        self.comparison_data = self.memo_by_params(dataset)
        repertoire_count = dataset.get_example_count()
        comparison_result = np.zeros([repertoire_count, repertoire_count])
        repertoire_identifiers = dataset.get_repertoire_ids()

        for index1 in range(repertoire_count):
            repertoire_vector_1 = self.comparison_data.get_repertoire_vector(repertoire_identifiers[index1])
            for index2 in range(index1, repertoire_count):
                repertoire_vector_2 = self.comparison_data.get_repertoire_vector(repertoire_identifiers[index2])
                comparison_result[index1, index2] = comparison_fn(repertoire_vector_1, repertoire_vector_2)
                comparison_result[index2, index1] = comparison_result[index1, index2]

        comparison_df = pd.DataFrame(comparison_result, columns=repertoire_identifiers, index=repertoire_identifiers)

        return comparison_df

    def prepare_paralellization_arguments(self, repertoire_count: int, repertoire_identifiers: list, comparison_result):

        arguments = []

        for index1 in range(repertoire_count):
            comparison_result[index1, index1] = 1
            rep1 = repertoire_identifiers[index1]
            for index2 in range(index1+1, repertoire_count):
                rep2 = repertoire_identifiers[index2]
                arguments.append((rep1, rep2))

        return arguments
