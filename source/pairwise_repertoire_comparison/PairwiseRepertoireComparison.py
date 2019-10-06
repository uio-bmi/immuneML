import os
from shutil import copyfile

import numpy as np
import pandas as pd

from source.caching.CacheHandler import CacheHandler
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

        comparison_data = self.add_files_to_cache(comparison_data, dataset)

        return comparison_data

    def add_files_to_cache(self, comparison_data: ComparisonData, dataset: RepertoireDataset) -> ComparisonData:

        cache_paths = []

        for index, batch_path in enumerate(comparison_data.batch_paths):
            cache_paths.append(CacheHandler.get_file_path() + "dataset_{}_batch_{}.csv".format(dataset.identifier, index))
            copyfile(batch_path, cache_paths[-1])

        comparison_data.batch_paths = cache_paths

        return comparison_data

    def prepare_caching_params(self, dataset: RepertoireDataset):
        return (
            ("dataset_identifier", dataset.identifier),
            ("item_attributes", self.item_columns)
        )

    def memo_by_params(self, dataset: RepertoireDataset):
        # TODO: refactor this to be immune to removing the cache halfway through repertoire comparison
        comparison_data = CacheHandler.memo_by_params(self.prepare_caching_params(dataset), lambda: self.create_comparison_data(dataset))
        if all(os.path.isfile(path) for path in comparison_data.batch_paths):
            return comparison_data
        else:
            return self.create_comparison_data(dataset)

    def compare_repertoires(self, dataset: RepertoireDataset, comparison_fn):
        comparison_data = self.memo_by_params(dataset)
        repertoire_count = dataset.get_example_count()
        comparison_result = np.zeros([repertoire_count, repertoire_count])

        repertoire_identifiers = []

        for index1, repertoire1 in enumerate(dataset.get_data()):
            for index2, repertoire2 in enumerate(dataset.get_data()):

                if index1 > index2:
                    continue

                rep1 = comparison_data.get_repertoire_vector(index1+1)
                rep2 = comparison_data.get_repertoire_vector(index2+1)

                comparison_result[index1, index2] = comparison_fn(rep1, rep2)
                comparison_result[index2, index1] = comparison_result[index1, index2]

                if index1 == 0:
                    repertoire_identifiers.append(repertoire2.identifier)

        comparison_df = pd.DataFrame(comparison_result, columns=repertoire_identifiers, index=repertoire_identifiers)

        return comparison_df
