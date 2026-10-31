import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.LabelDist import LabelDist
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestLabelDist(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_generate(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "label_dist/")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(50, {10: 1.},
                                                                   {"l1": {"a": 0.5, "b": 0.5},
                                                                    "l2": {"x": 0.2, "y": 0.3, "z": 0.5}},
                                                                   path / "dataset")

        report = LabelDist.build_object(labels=["l1", "l2"], name="label_dist")
        report.dataset = dataset
        report.result_path = path / "result"

        result = report._generate()

        self.assertTrue(os.path.isfile(path / "result/label_dist_label_distributions.html"))
        self.assertEqual(1, len(result.output_tables))

        counts = pd.read_csv(path / "result/label_distribution.csv")
        self.assertListEqual(list(counts.columns), ["label", "value", "count"])
        metadata = dataset.get_metadata(["l1", "l2"], return_df=True)
        for label in ["l1", "l2"]:
            label_counts = counts[counts["label"] == label]
            expected = metadata[label].astype(str).value_counts().to_dict()
            self.assertDictEqual(dict(zip(label_counts["value"].astype(str), label_counts["count"])), expected)
            # most frequent first within each label
            self.assertListEqual(list(label_counts["count"]), sorted(label_counts["count"], reverse=True))

        shutil.rmtree(path)
