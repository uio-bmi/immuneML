import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
import yaml

from source.caching.CacheType import CacheType
from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.LogisticRegression import LogisticRegression
from source.reports.ReportResult import ReportResult
from source.reports.ml_reports.MotifSeedRecovery import MotifSeedRecovery
from source.util.PathBuilder import PathBuilder


class TestMotifSeedRecovery(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_lr_model(self, path):
        # dummy logistic regression with 100 observations with 20 features belonging to 2 classes
        dummy_lr = LogisticRegression()
        dummy_lr.fit_by_cross_validation(EncodedData(np.random.rand(100, 5), {"l1": [i % 2 for i in range(0, 100)]}), number_of_splits=2,
                                         label_name="l1")

        # Change coefficients to values 1-20
        dummy_lr.models["l1"].coef_ = np.array(list(range(0, 5))).reshape(1, -1)

        with open(path  / "ml_details.yaml", "w") as file:
            yaml.dump({"l1": {"feature_names": ["AAA", "AAC", "CKJ", "KSA", "AKJ"]}},
                      file)

        return dummy_lr

    def _create_report(self, path):
        report = MotifSeedRecovery.build_object(**{"implanted_motifs_per_label": {
                "l1": {"seeds": ["AAA", "A/AA"],
                       "hamming_distance": False,
                       "gap_sizes": [1]}}})

        report.method = self._create_dummy_lr_model(path)
        report.label = "l1"
        report.result_path = path
        report.train_dataset = Dataset()
        report.train_dataset.encoded_data = EncodedData(examples=np.zeros((1, 5)), labels={"l1": [1]}, feature_names=["AAA", "AAC", "CKJ", "KSA", "AKJ"])

        return report

    def test_generate(self):
        path = EnvironmentSettings.root_path  / "test/tmp/motifseedrecovery/"
        PathBuilder.build(path)

        report = self._create_report(path)

        # Running the report
        report.check_prerequisites()
        result = report.generate()

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(result.output_tables[0].path, path / "motif_seed_recovery.csv")
        self.assertEqual(result.output_figures[0].path, path / "motif_seed_recovery.html")

        # Actual tests
        self.assertTrue(os.path.isfile(path / "motif_seed_recovery.csv"))
        self.assertTrue(os.path.isfile(path / "motif_seed_recovery.html"))

        written_data = pd.read_csv(path / "motif_seed_recovery.csv")

        self.assertListEqual(list(written_data.columns), ["features", "max_seed_overlap", "coefficients"])
        self.assertListEqual(list(written_data["coefficients"]), [i for i in range(5)])
        self.assertListEqual(list(written_data["features"]), ["AAA", "AAC", "CKJ", "KSA", "AKJ"])
        self.assertListEqual(list(written_data["max_seed_overlap"]), [3, 2, 0, 1, 1])

        shutil.rmtree(path)

    def test_overlap(self):
        report = MotifSeedRecovery.build_object(**{"implanted_motifs_per_label": {
            "l1": {"seeds": ["AAA", "A/AA"],
                   "hamming_distance": False,
                   "gap_sizes": [1]}}})
        report.label = "l1"

        self.assertEqual(report.identical_overlap(seed="AAA", feature="AAA"), 3)
        self.assertEqual(report.identical_overlap(seed="AAA", feature="AAx"), 0)

        self.assertEqual(report.identical_overlap(seed="AA/A", feature="AAxA"), 3)
        self.assertEqual(report.identical_overlap(seed="AA/A", feature="AAxx"), 0)

        self.assertEqual(report.hamming_overlap(seed="AAA", feature="AAA"), 3)
        self.assertEqual(report.hamming_overlap(seed="AAA", feature="AAx"), 2)
        self.assertEqual(report.hamming_overlap(seed="AAA", feature="xAx"), 1)

        self.assertEqual(report.hamming_overlap(seed="AA/A", feature="AAxA"), 3)
        self.assertEqual(report.hamming_overlap(seed="AA/A", feature="AAxx"), 2)


        self.assertEqual(report.max_overlap_sliding(seed="AAA", feature="xAAAx", overlap_fn=report.identical_overlap), 3)
        self.assertEqual(report.max_overlap_sliding(seed="AAA", feature="xAAxx", overlap_fn=report.identical_overlap), 0)
        self.assertEqual(report.max_overlap_sliding(seed="AAA", feature="AAxx", overlap_fn=report.identical_overlap), 2)

        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xAAxAx", overlap_fn=report.identical_overlap), 3)
        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xAAxxx", overlap_fn=report.identical_overlap), 1)

        self.assertEqual(report.max_overlap_sliding(seed="AAA", feature="xAAAx", overlap_fn=report.hamming_overlap), 3)
        self.assertEqual(report.max_overlap_sliding(seed="AAA", feature="xAAxx", overlap_fn=report.hamming_overlap), 2)
        self.assertEqual(report.max_overlap_sliding(seed="AAA", feature="xxAxx", overlap_fn=report.hamming_overlap), 1)

        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xAAxAx", overlap_fn=report.hamming_overlap), 3)
        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xAAxxx", overlap_fn=report.hamming_overlap), 2)

    def test_overlap_gap(self):
        report = MotifSeedRecovery.build_object(**{"implanted_motifs_per_label": {
            "l1": {"seeds": ["AAA", "A/AA"],
                   "hamming_distance": False,
                   "gap_sizes": [0, 5]}}})
        report.label = "l1"

        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xAAAx", overlap_fn=report.identical_overlap), 3)
        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xAAxxxxxAx", overlap_fn=report.identical_overlap), 3)
        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xxxxxxxxAAxxxxxxx", overlap_fn=report.identical_overlap), 0)
        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xAAxxxxxxx", overlap_fn=report.identical_overlap), 1)

        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xAAAx", overlap_fn=report.hamming_overlap), 3)
        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xAAxx", overlap_fn=report.hamming_overlap), 2)
        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xAAxxxxxAx", overlap_fn=report.hamming_overlap), 3)
        self.assertEqual(report.max_overlap_sliding(seed="AA/A", feature="xAAxxxxxxx", overlap_fn=report.hamming_overlap), 2)
