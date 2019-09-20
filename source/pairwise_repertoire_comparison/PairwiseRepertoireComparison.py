import itertools

import numpy as np

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from source.util.PathBuilder import PathBuilder


class PairwiseRepertoireComparison:

    def __init__(self, matching_columns: list, item_columns: list, path: str, batch_size: int, extract_items_fn):
        self.matching_columns = matching_columns
        self.item_columns = item_columns
        self.path = path
        PathBuilder.build(path)
        self.batch_size = batch_size
        self.extract_items_fn = extract_items_fn

    def create_comparison_data(self, dataset: RepertoireDataset) -> ComparisonData:

        comparison_data = ComparisonData(dataset.get_example_count(), self.matching_columns, self.item_columns, self.path, self.batch_size)

        for index, repertoire in enumerate(dataset.get_data()):
            comparison_data.process_repertoire(repertoire, index+1, self.extract_items_fn)

        return comparison_data

    def compare_repertoires(self, dataset: RepertoireDataset, comparison_fn):
        comparison_data = self.create_comparison_data(dataset)
        repertoire_count = dataset.get_example_count()
        comparison_result = np.zeros([repertoire_count, repertoire_count])

        for repertoire_pair in itertools.product(range(repertoire_count), repeat=2):
            rep1 = comparison_data.get_repertoire_vector(repertoire_pair[0]+1)
            rep2 = comparison_data.get_repertoire_vector(repertoire_pair[1]+1)

            comparison_result[repertoire_pair[0], repertoire_pair[1]] = comparison_fn(rep1, rep2)

        return comparison_result
