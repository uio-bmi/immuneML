import os
import shutil
from unittest import TestCase

import pandas as pd

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.data_reports.GLIPH2Exporter import GLIPH2Exporter
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from source.util.PathBuilder import PathBuilder


class TestGLIPH2Exporter(TestCase):
    def test_generate(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path + "gliph2_export/")
        dataset = RandomDatasetGenerator.generate_receptor_dataset(10, {3: 1}, {2: 1}, {"epitope": {"ep1": 0.4, "ep2": 0.6}}, path)
        report_result = GLIPH2Exporter(dataset, path + "result/", "somename", "epitope").generate_report()

        self.assertEqual(1, len(report_result.output_tables))
        self.assertTrue(os.path.isfile(report_result.output_tables[0].path))

        df = pd.read_csv(report_result.output_tables[0].path, sep="\t")
        self.assertTrue(all(col in ["CDR3b", "TRBV", "TRBJ", "CDR3a", "subject:condition", "count"] for col in df.columns))
        self.assertEqual(10, df.shape[0])

        shutil.rmtree(path)
