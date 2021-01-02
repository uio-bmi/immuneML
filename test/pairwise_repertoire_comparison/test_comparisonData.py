import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from source.caching.CacheType import CacheType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from source.pairwise_repertoire_comparison.ComparisonDataBatch import ComparisonDataBatch
from source.util.PathBuilder import PathBuilder


class TestComparisonData(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_comparison_data(self, path: str):
        comparison_data = ComparisonData(repertoire_ids=["1", "2", "3", "4", "5", "6"], comparison_attributes=["col1", "col2"],
                                         sequence_batch_size=3, path=path)

        comparison_data.tmp_batch_paths = []
        batch0 = {("a", 1): {"1": 1}, ("b", 2): {"2": 1}, ("c", 3): {"3": 1}}
        comparison_data.store_tmp_batch(batch0, 0)

        batch1 = {("d", 4): {"1": 1}, ("e", 5): {"5": 1}}
        comparison_data.store_tmp_batch(batch1, 1)
        comparison_data.matching_columns = ["col1", "col2"]
        comparison_data.item_count = 5
        df1 = pd.DataFrame({"1": [1, 0, 0], "2": [0, 1, 0], "3": [0, 0, 1], "4": [0, 0, 0],
                            "5": [0, 0, 0], "6": [0, 0, 0]})
        comp_data_batch = ComparisonDataBatch(matrix=df1.values, items=[("a", 1), ("b", 2), ("c", 3)],
                                              repertoire_index_mapping={"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 7},
                                              path=path, identifier=0)
        comp_data_batch.store()
        comparison_data.batches.append(comp_data_batch)
        df2 = pd.DataFrame({"1": [1, 0], "2": [0, 0], "3": [0, 0], "4": [0, 0],
                            "5": [0, 1], "6": [0, 0]})
        comp_data_batch = ComparisonDataBatch(matrix=df2.values, items=[("d", 4), ("e", 5)],
                                              repertoire_index_mapping={"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 7},
                                              path=path, identifier=1)
        comp_data_batch.store()
        comparison_data.batches.append(comp_data_batch)
        return comparison_data

    def test_get_repertoire_vector(self):
        path = EnvironmentSettings.tmp_test_path / "pairwisecomp_comparisondata/"
        PathBuilder.build(path)

        comparison_data = self.create_comparison_data(path=path)

        rep_vector = comparison_data.get_repertoire_vector("2")
        self.assertTrue(np.array_equal(rep_vector, [0, 1, 0, 0, 0]))

        shutil.rmtree(path)

    def test_get_feature_vector(self):
        path = EnvironmentSettings.tmp_test_path / "pairwisecomp_comparisondata_item_vector/"
        PathBuilder.build(path)

        comparison_data = self.create_comparison_data(path=path)

        item_vector = comparison_data.get_item_vector(3)
        self.assertTrue(np.array_equal(item_vector, np.array([1, 0, 0, 0, 0, 0], dtype=object)))

        shutil.rmtree(path)

    def test_get_batches(self):

        path = EnvironmentSettings.tmp_test_path / "get_batches_comp_data/"
        PathBuilder.build(path)

        comparison_data = self.create_comparison_data(path=path)
        index = 0
        items = 0
        for batch in comparison_data.get_batches():
            index += 1
            self.assertTrue(batch.shape[1] == len(["1", "2", "3", "4", "5", "6"]))
            items += batch.shape[0]

        self.assertEqual(2, index)
        self.assertEqual(5, items)

        for index, batch in enumerate(comparison_data.get_batches(columns=["1"])):
            self.assertEqual(3 - index, batch.shape[0])

        shutil.rmtree(path)

    def test_filter_existing_items(self):

        path = EnvironmentSettings.tmp_test_path / "comparison_rep_filter_existing/"
        PathBuilder.build(path)

        comparison_data = self.create_comparison_data(path=path)
        unique_items = comparison_data.filter_existing_items([("f", 6), ("g", 7), ("a", 1)], "6")

        self.assertEqual(2, len(unique_items))
        self.assertTrue("f" in [unique_items[0][0], unique_items[1][0]])
        self.assertTrue("g" in [unique_items[0][0], unique_items[1][0]])

        comparison_data = self.create_comparison_data(path=path)
        comparison_data.batch_paths = comparison_data.tmp_batch_paths
        unique_items = comparison_data.filter_existing_items([("f", 6), ("g", 7), ("a", 1), ("b", 2), ("c", 3), ("d", 4)], "6")

        self.assertEqual(2, len(unique_items))
        self.assertTrue("f" in [unique_items[0][0], unique_items[1][0]])
        self.assertTrue("g" in [unique_items[0][0], unique_items[1][0]])

        shutil.rmtree(path)
