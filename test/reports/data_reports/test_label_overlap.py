import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.data_reports.LabelOverlap import LabelOverlap
from immuneML.util.PathBuilder import PathBuilder


class TestLabelOverlap(TestCase):

    def setUp(self) -> None:
        os.environ["cache_type"] = "test"
        self.path = PathBuilder.remove_old_and_build(f'{EnvironmentSettings.tmp_test_path}/label_overlap/')
        self.create_dummy_dataset()

    def create_dummy_dataset(self):
        # Create a dataset with dummy labels
        labels = {
            "subject_id": ["s1", "s1", "s2", "s2", "s3"],
            "disease": ["healthy", "healthy", "sick", "sick", "healthy"],
            "age_group": ["young", "young", "old", "old", "young"]
        }

        label_config = LabelConfiguration()
        label_config.add_label("subject_id", values=["s1", "s2", "s3"])
        label_config.add_label("disease", values=["healthy", "sick"])
        label_config.add_label("age_group", values=["young", "old"])

        metadata_filepath = self.path / "metadata.csv"
        pd.DataFrame(labels).to_csv(metadata_filepath, index=False)

        self.dataset = RepertoireDataset(labels=labels, metadata_file=metadata_filepath)
        self.dataset.metadata_file = metadata_filepath

    def test_generate(self):
        # Test report generation with valid labels
        report = LabelOverlap(dataset=self.dataset,
                              result_path=self.path / "results/",
                              name="label_overlap_report",
                              column_label="disease",
                              row_label="age_group")

        # Check prerequisites
        self.assertTrue(report.check_prerequisites())

        # Generate report
        result = report._generate()

        # Check if files were created
        self.assertTrue(os.path.isfile(self.path / "results/label_overlap.csv"))
        self.assertTrue(os.path.isfile(self.path / "results/label_overlap.html"))

        # Check CSV content
        df = pd.read_csv(self.path / "results/label_overlap.csv", index_col=0)
        self.assertEqual(df.shape, (2, 2))  # 2x2 matrix for age_group vs disease
        self.assertEqual(df.loc["young", "healthy"], 3)  # 3 young healthy subjects
        self.assertEqual(df.loc["old", "sick"], 2)  # 2 old sick subjects

        # Check report result
        self.assertEqual(len(result.output_figures), 1)
        self.assertEqual(len(result.output_tables), 1)
        self.assertIsNotNone(result.info)

    def test_invalid_labels(self):
        # Test with invalid label names
        report = LabelOverlap(dataset=self.dataset,
                              result_path=self.path / "results/",
                              name="label_overlap_report",
                              column_label="invalid_label",
                              row_label="disease")

        self.assertFalse(report.check_prerequisites())

    def tearDown(self) -> None:
        shutil.rmtree(self.path.parent, ignore_errors=True)
