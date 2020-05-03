import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from source.util.PathBuilder import PathBuilder


class TestComparisonData(TestCase):

    def create_comparison_data(self, path: str):
        comparison_data = ComparisonData(repertoire_ids=["1", "2", "3", "4", "5", "6"], comparison_attributes=["col1", "col2"],
                                         pool_size=4, batch_size=3, path=path)

        comparison_data.tmp_batch_paths = [path + "batch_0.csv", path + "batch_1.csv"]
        batch0 = {("a", 1): {"1": 1}, ("b", 2): {"2": 1}, ("c", 3): {"3": 1}}
        comparison_data.store_tmp_batch(batch0, 0)

        batch1 = {("d", 4): {"1": 1}, ("e", 5): {"5": 1}}
        comparison_data.store_tmp_batch(batch1, 1)
        comparison_data.matching_columns = ["col1", "col2"]
        comparison_data.item_count = 5
        df1 = pd.DataFrame({"1": [1, 0, 0], "2": [0, 1, 0], "3": [0, 0, 1], "4": [0, 0, 0],
                            "5": [0, 0, 0], "6": [0, 0, 0]})
        comparison_data.batches.append({"matrix": df1.values, "row_names": [("a", 1), ("b", 2), ("c", 3)],
                                        "col_name_index": {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 7}})
        df1.to_csv(path + "b01.csv", index=False)
        df2 = pd.DataFrame({"1": [1, 0], "2": [0, 0], "3": [0, 0], "4": [0, 0],
                            "5": [0, 1], "6": [0, 0]})
        df2.to_csv(path + "b02.csv", index=False)
        comparison_data.batches.append({"matrix": df2.values, "row_names": [("d", 4), ("e", 5)],
                                        "col_name_index": {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 7}})
        comparison_data.batch_paths = [path + "b01.csv", path + "b02.csv"]
        return comparison_data

    def test_build_abundance_matrix(self):
        expected_abundance_matrix = np.array([[1, 4], [1, 6], [1, 3], [1, 6]])

        comparison_data = ComparisonData(repertoire_ids=["rep_0", "rep_1", "rep_2", "rep_3"],
                                         comparison_attributes=["amino_acid_sequence"], pool_size=4, batch_size=2, path="")
        comparison_data.batches = [{'matrix': np.array([[1., 0., 0., 0.],
                                                        [1., 1., 0., 0.]]),
                                    'row_names': [('GGG',), ('III',)],
                                    'col_name_index': {'rep_0': 0, 'rep_1': 1, 'rep_2': 2, 'rep_3': 3}},
                                   {'matrix': np.array([[1., 1., 0., 1.],
                                                        [1., 1., 1., 1.]]),
                                    'row_names': [('LLL',), ('MMM',)],
                                    'col_name_index': {'rep_0': 0, 'rep_1': 1, 'rep_2': 2, 'rep_3': 3}},
                                   {'matrix': np.array([[0., 1., 0., 0.],
                                                        [0., 1., 0., 1.]]),
                                    'row_names': [('DDD',), ('EEE',)],
                                    'col_name_index': {'rep_0': 0, 'rep_1': 1, 'rep_2': 2, 'rep_3': 3}},
                                   {'matrix': np.array([[0., 1., 1., 1.],
                                                        [0., 0., 1., 1.]]),
                                    'row_names': [('FFF',), ('CCC',)],
                                    'col_name_index': {'rep_0': 0, 'rep_1': 1, 'rep_2': 2, 'rep_3': 3}},
                                   {'matrix': np.array([[0., 0., 0., 1.]]),
                                    'row_names': [('AAA',)],
                                    'col_name_index': {'rep_0': 0, 'rep_1': 1, 'rep_2': 2, 'rep_3': 3}}]
        comparison_data.item_count = 9

        p_value = 0.4
        sequence_p_value_indices = np.array([1., 0.3333333333333334, 1., 1., 1., 1., 1., 0.3333333333333334, 1.]) < p_value

        abundance_matrix = comparison_data.build_abundance_matrix(["rep_0", "rep_1", "rep_2", "rep_3"], sequence_p_value_indices)

        self.assertTrue(np.array_equal(expected_abundance_matrix, abundance_matrix))

    def test_find_label_associated_sequence_p_values(self):

        path = EnvironmentSettings.tmp_test_path + "comparisondatafindlabelassocseqpvalues/"
        PathBuilder.build(path)

        repertoires = [Repertoire.build_from_sequence_objects([ReceptorSequence()], path, {
            "l1": val, "donor": donor
        }) for val, donor in zip([True, True, False, False], ["rep_0", "rep_1", "rep_2", "rep_3"])]

        col_name_index = {repertoires[index].identifier: index for index in range(len(repertoires))}

        comparison_data = ComparisonData(repertoire_ids=[repertoire.identifier for repertoire in repertoires],
                                         comparison_attributes=["amino_acid_sequence"], pool_size=4, path="")
        comparison_data.batches = [{'matrix': np.array([[1., 0., 0., 0.],
                                                        [1., 1., 0., 0.]]),
                                    'row_names': [('GGG',), ('III',)],
                                    'col_name_index': col_name_index},
                                   {'matrix': np.array([[1., 1., 0., 1.],
                                                        [1., 1., 1., 1.]]),
                                    'row_names': [('LLL',), ('MMM',)],
                                    'col_name_index': col_name_index},
                                   {'matrix': np.array([[0., 1., 0., 0.],
                                                        [0., 1., 0., 1.]]),
                                    'row_names': [('DDD',), ('EEE',)],
                                    'col_name_index': col_name_index},
                                   {'matrix': np.array([[0., 1., 1., 1.],
                                                        [0., 0., 1., 1.]]),
                                    'row_names': [('FFF',), ('CCC',)],
                                    'col_name_index': col_name_index},
                                   {'matrix': np.array([[0., 0., 0., 1.]]),
                                    'row_names': [('AAA',)],
                                    'col_name_index':col_name_index}]
        p_values = comparison_data.find_label_associated_sequence_p_values(repertoires, "l1", [True, False])

        self.assertTrue(np.allclose([np.nan, 0.3333333333333334, 1., 1., np.nan, 1., 1., 0.3333333333333334, np.nan], p_values,
                                    equal_nan=True))

        shutil.rmtree(path)

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
        self.assertTrue(np.array_equal(item_vector, np.array([1, 0, 0, 0, 0, 0], dtype=object)))

        shutil.rmtree(path)

    def test_get_batches(self):

        path = EnvironmentSettings.tmp_test_path + "get_batches_comp_data/"
        PathBuilder.build(path)

        comparison_data = self.create_comparison_data(path=path)
        index = 0
        items = 0
        for batch in comparison_data.get_batches():
            index += 1
            self.assertTrue(all(col in batch["col_name_index"] for col in ["1", "2", "3", "4", "5", "6"]))
            items += batch["matrix"].shape[0]

        self.assertEqual(2, index)
        self.assertEqual(5, items)

        for index, batch in enumerate(comparison_data.get_batches(columns=["1"])):

            self.assertEqual(3-index, batch.shape[0])

        shutil.rmtree(path)

    def test_filter_existing_items(self):

        path = EnvironmentSettings.tmp_test_path + "comparison_rep_filter_existing/"
        PathBuilder.build(path)

        comparison_data = self.create_comparison_data(path=path)
        comparison_data.batch_paths = comparison_data.tmp_batch_paths
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
