import shutil
from unittest import TestCase

import pandas as pd

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.ReportOutput import ReportOutput
from source.reports.hyperparameter_reports.ReferenceSequenceOverlap import ReferenceSequenceOverlap
from source.util.PathBuilder import PathBuilder


class TestReferenceSequenceOverlap(TestCase):
    def test__compute_model_overlap(self):

        path = PathBuilder.build(EnvironmentSettings.tmp_test_path + "ref_sequence_overlap/")

        ref_path = path + "reference.csv"
        pd.DataFrame({"sequence_aas": ["AAA", "ACC", 'TTT', "ACA"], "v_genes": ["V1", "V1", "V1", "V1"], "j_genes": ["J1", "J1", "J1", "J1"]}).to_csv(ref_path, index=False)
        model_path = path + "model.csv"
        pd.DataFrame({"sequence_aas": ["AAA", "ACC", "TTT", "ATA", "TAA"], "v_genes": ["V1", "V1", "V1", "V1", "V1"]}).to_csv(model_path, index=False)

        report = ReferenceSequenceOverlap(result_path=path, reference_path=ref_path, comparison_attributes=['sequence_aas', 'v_genes'])

        class Enc:
            def __init__(self, relevant_sequence_csv_path):
                self.relevant_sequence_csv_path = relevant_sequence_csv_path

        encoder = Enc(model_path)

        figure, data = report._compute_model_overlap(path+"figure.pdf", path+"overlap.csv", encoder, "sample name")

        self.assertTrue(isinstance(figure, ReportOutput))
        self.assertTrue(isinstance(data, ReportOutput))

        self.assertEqual(3, pd.read_csv(data.path).shape[0])
        self.assertEqual(2, pd.read_csv(data.path).shape[1])

        shutil.rmtree(path)
