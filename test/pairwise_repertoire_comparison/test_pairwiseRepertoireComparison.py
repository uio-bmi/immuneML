import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML.IO.ml_method.UtilIO import UtilIO
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.pairwise_repertoire_comparison.PairwiseRepertoireComparison import PairwiseRepertoireComparison
from immuneML.util import DistanceMetrics
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestPairwiseRepertoireComparison(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dataset(self, path: str) -> RepertoireDataset:
        repertoires, metadata = RepertoireBuilder.build([["A", "B"], ["D"], ["E", "F"], ["B", "C"], ["A", "D"]], path)
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        return dataset

    def test_compare_repertoires(self):

        path = EnvironmentSettings.tmp_test_path / "pairwise_comparison_reps/"
        PathBuilder.build(path)

        dataset = self.create_dataset(path)

        comparison = PairwiseRepertoireComparison(["sequence_aa"], ["sequence_aa"], path, 4)

        # comparison_fn = lambda rep1, rep2, tmp_vector: np.sum(np.logical_and(rep1, rep2)) / np.sum(np.logical_or(rep1, rep2))
        comparison_fn = DistanceMetrics.jaccard

        result = comparison.compare_repertoires(dataset, comparison_fn)

        self.assertTrue(np.array_equal(result, np.array([[0., 1., 1., 0.6666666666666667, 0.6666666666666667],
                                                         [1., 0., 1., 1., 0.5],
                                                         [1., 1., 0., 1., 1.],
                                                         [0.6666666666666667, 1., 1., 0., 1.],
                                                         [0.6666666666666667, 0.5, 1., 1., 0.]])))

        shutil.rmtree(path)

    def test_comparison_data_io(self):
        path = EnvironmentSettings.tmp_test_path / "comparison_data_io/"
        PathBuilder.build(path)

        dataset = self.create_dataset(path)

        comparison = PairwiseRepertoireComparison(["sequence_aas"], ["sequence_aas"], path, 4)

        comparison_data = comparison.create_comparison_data(dataset)

        export_path = UtilIO.export_comparison_data(comparison_data, path)

        imported_comparison_data = UtilIO.import_comparison_data(path)

        self.assertEqual(comparison_data.repertoire_ids, imported_comparison_data.repertoire_ids)
        self.assertEqual(comparison_data.comparison_attributes, imported_comparison_data.comparison_attributes)
        self.assertEqual(comparison_data.item_count, imported_comparison_data.item_count)
        self.assertEqual(comparison_data.item_count, imported_comparison_data.item_count)

        shutil.rmtree(path)



