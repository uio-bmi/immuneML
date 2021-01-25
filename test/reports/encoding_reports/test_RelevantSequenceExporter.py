import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.encoding_reports.RelevantSequenceExporter import RelevantSequenceExporter
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestRelevantSequenceExporter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_generate(self):
        path = EnvironmentSettings.tmp_test_path / "relevant_sequence_exporter/"
        PathBuilder.build(path)

        df = pd.DataFrame({"v_genes": ["TRBV1-1", "TRBV1-1"], 'j_genes': ["TRBJ1-1", "TRBJ1-2"], "sequence_aas": ['ACCF', "EEFG"]})
        df.to_csv(path / 'sequences.csv', index=False)

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(2, {2: 1}, {4: 1}, {}, path / "data")
        dataset.encoded_data = EncodedData(examples=None, info={'relevant_sequence_path': path / 'sequences.csv'}, encoding="SequenceAbundanceEncoder")

        report_result = RelevantSequenceExporter(dataset, path / "result", 'somename').generate_report()

        self.assertEqual(1, len(report_result.output_tables))
        self.assertTrue(os.path.isfile(report_result.output_tables[0].path))

        self.assertTrue(all(col in ["v_call", "j_call", "cdr3_aa"] for col in pd.read_csv(report_result.output_tables[0].path).columns))

        shutil.rmtree(path)
