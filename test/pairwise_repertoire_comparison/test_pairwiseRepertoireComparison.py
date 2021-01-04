import os
import shutil
from unittest import TestCase

import numpy as np

from source.IO.ml_method.UtilIO import UtilIO
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.pairwise_repertoire_comparison.PairwiseRepertoireComparison import PairwiseRepertoireComparison
from source.util import DistanceMetrics
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


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

        comparison = PairwiseRepertoireComparison(["sequence_aas"], ["sequence_aas"], path, 4)

        # comparison_fn = lambda rep1, rep2, tmp_vector: np.sum(np.logical_and(rep1, rep2)) / np.sum(np.logical_or(rep1, rep2))
        comparison_fn = DistanceMetrics.jaccard

        result = comparison.compare_repertoires(dataset, comparison_fn)

        self.assertTrue(np.array_equal(result, np.array([[1., 0., 0., 0.3333333333333333, 0.3333333333333333],
                                                         [0., 1., 0., 0., 0.5],
                                                         [0., 0., 1., 0., 0.],
                                                         [0.3333333333333333, 0., 0., 1., 0.],
                                                         [0.3333333333333333, 0.5, 0., 0., 1.]])))

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



