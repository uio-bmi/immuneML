import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from source.util.PathBuilder import PathBuilder


class TestComparisonData(TestCase):

    def create_comparison_data(self, path: str):
        comparison_data = ComparisonData(repertoire_ids=["1", "2", "3", "4", "5", "6"],
                                         matching_columns=["col1", "col2"], item_columns=["col1", "col2"],
                                         path=path, batch_size=3)
        df1 = pd.DataFrame({"col1": ["a", "b", "c"], "col2": [1, 2, 3], "rep_1": [1, 0, 0], "rep_2": [0, 1, 0], "rep_3": [0, 0, 1],
                            "rep_4": [0, 0, 0], "rep_5": [0, 0, 0], "rep_6": [0, 0, 0]})
        df1.to_csv(path + "batch1.csv", index=False)
        df2 = pd.DataFrame({"col1": ["d", "e"], "col2": [4, 5], "rep_1": [0, 0], "rep_2": [0, 0], "rep_3": [0, 0], "rep_4": [1, 0],
                            "rep_5": [0, 1], "rep_6": [0, 0]})
        df2.to_csv(path + "batch2.csv", index=False)
        comparison_data.batch_paths = [path + "batch1.csv", path + "batch2.csv"]

        comparison_data.item_count = 5
        comparison_data.matching_columns = ["col1", "col2"]
        return comparison_data

    def test_get_repertoire_vector(self):
        path = EnvironmentSettings.tmp_test_path + "pairwisecomp_comparisondata/"
        PathBuilder.build(path)

        comparison_data = self.create_comparison_data(path=path)

        rep_vector = comparison_data.get_repertoire_vector("2")
        self.assertTrue(np.array_equal(rep_vector, [0, 1, 0, 0, 0]))

        shutil.rmtree(path)

    def test_get_item_vector(self):
        path = EnvironmentSettings.tmp_test_path + "pairwisecomp_comparisondata_item_vector/"
        PathBuilder.build(path)

        comparison_data = self.create_comparison_data(path=path)

        item_vector = comparison_data.get_item_vector(3)
        self.assertTrue(np.array_equal(item_vector, np.array(["d", 4, 0, 0, 0, 1, 0, 0], dtype=object)))

        shutil.rmtree(path)

    def test_get_batches(self):

        path = EnvironmentSettings.tmp_test_path + "get_batches_comp_data/"
        PathBuilder.build(path)

        comparison_data = self.create_comparison_data(path=path)
        index = 0
        items = 0
        for batch in comparison_data.get_batches():
            index += 1
            self.assertTrue(all(col in batch.columns for col in ["col1", "col2", "rep_1", "rep_2", "rep_3", "rep_4", "rep_5"]))
            items += batch.shape[0]

        self.assertEqual(2, index)
        self.assertEqual(5, items)

        for batch in comparison_data.get_batches(columns=["rep_1", "col1"]):
            self.assertTrue(all(col in batch.columns for col in ["rep_1", "col1"]))
            self.assertEqual(2, len(batch.columns))

        shutil.rmtree(path)

    def test_filter_existing_items(self):

        path = EnvironmentSettings.tmp_test_path + "comparison_rep_filter_existing/"
        PathBuilder.build(path)

        comparison_data = self.create_comparison_data(path=path)
        unique_items = comparison_data.filter_existing_items(pd.DataFrame({"col1": ["f", "g", "a"], "col2": [6, 7, 1]}), "6")

        self.assertEqual(2, unique_items.shape[0])
        self.assertTrue("f" in unique_items["col1"].values)
        self.assertTrue("g" in unique_items["col1"].values)

        comparison_data = self.create_comparison_data(path=path)
        unique_items = comparison_data.filter_existing_items(pd.DataFrame({"col1": ["f", "g", "a", "b", "c", "d"],
                                                                           "col2": [6, 7, 1, 2, 3, 4]}), "6")

        self.assertEqual(2, unique_items.shape[0])
        self.assertTrue("f" in unique_items["col1"].values)
        self.assertTrue("g" in unique_items["col1"].values)

        shutil.rmtree(path)

    def test_add_items_for_repertoire(self):
        path = EnvironmentSettings.tmp_test_path + "comparison_rep_add_items/"
        PathBuilder.build(path)

        comparison_data = self.create_comparison_data(path=path)
        items = pd.DataFrame({"col1": ["x", "y"], "col2": [10, 11]})

        comparison_data.add_items_for_repertoire(items, "6")

        self.assertEqual(7, comparison_data.item_count)
        self.assertEqual(3, len(comparison_data.batch_paths))
        self.assertEqual(2, sum(comparison_data.get_repertoire_vector("6")))
        self.assertEqual(1, sum(comparison_data.get_repertoire_vector("2")))

        shutil.rmtree(path)
