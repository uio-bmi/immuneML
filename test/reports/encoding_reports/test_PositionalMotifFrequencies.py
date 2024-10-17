import os
import shutil
import pandas as pd
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.reports.encoding_reports.PositionalMotifFrequencies import PositionalMotifFrequencies
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder


class TestPositionalMotifFrequencies(TestCase):
    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_encoded_data(self, path):
        sequences = [
            ReceptorSequence(
                sequence_aa="AACC",
                sequence_id="1",
                metadata={"l1": 1},
            ),
            ReceptorSequence(
                sequence_aa="AGDD",
                sequence_id="2",
                metadata={"l1": 1},
            ),
            ReceptorSequence(
                sequence_aa="AAEE",
                sequence_id="3",
                metadata={"l1": 1},
            ),
            ReceptorSequence(
                sequence_aa="AGFF",
                sequence_id="4",
                metadata={"l1": 1},
            ),
            ReceptorSequence(
                sequence_aa="CCCC",
                sequence_id="5",
                metadata={"l1": 2},
            ),
            ReceptorSequence(
                sequence_aa="DDDD",
                sequence_id="6",
                metadata={"l1": 2},
            ),
            ReceptorSequence(
                sequence_aa="EEEE",
                sequence_id="7",
                metadata={"l1": 2},
            ),
            ReceptorSequence(
                sequence_aa="FFFF",
                sequence_id="8",
                metadata={"l1": 2},
            ),
        ]

        PathBuilder.build(path)

        dataset = SequenceDataset.build_from_objects(
            sequences, PathBuilder.build(path / "data"), "d1"
        )

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2], positive_class=1)

        encoder = MotifEncoder.build_object(
            dataset,
            **{
                "min_positions": 1,
                "max_positions": 2,
                "min_precision": 0.9,
                "min_recall": 0.5,
                "min_true_positives": 1,
            }
        )

        encoded_dataset = encoder.encode(
            dataset,
            EncoderParams(
                result_path=path / "encoded_data/",
                label_config=lc,
                pool_size=2,
                learn_model=True,
                model={},
            ),
        )

        return encoded_dataset

    def test_generate(self):
        path = EnvironmentSettings.tmp_test_path / "positional_motif_frequencies/"
        PathBuilder.remove_old_and_build(path)

        encoded_dataset = self._create_dummy_encoded_data(path)

        report = PositionalMotifFrequencies.build_object(
            **{"dataset": encoded_dataset, "result_path": path,
               "motif_color_map": {1: "#66C5CC", 2: "#F6CF71", 3: "#F89C74"}}
        )

        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        self.assertTrue(os.path.isfile(path / "max_gap_size.html"))
        self.assertTrue(os.path.isfile(path / "total_gap_size.html"))
        self.assertTrue(os.path.isfile(path / "positional_motif_frequencies.html"))
        self.assertTrue(os.path.isfile(path / "max_gap_size_table.csv"))
        self.assertTrue(os.path.isfile(path / "total_gap_size_table.csv"))
        self.assertTrue(os.path.isfile(path / "positional_aa_counts.csv"))

        content = pd.read_csv(path / "max_gap_size_table.csv")
        self.assertEqual((list(content.columns))[1], "max_gap_size")
        self.assertEqual((list(content.columns))[2], "occurrence")

        content = pd.read_csv(path / "total_gap_size_table.csv")
        self.assertEqual((list(content.columns))[1], "total_gap_size")
        self.assertEqual((list(content.columns))[2], "occurrence")

        content = pd.read_csv(path / "positional_aa_counts.csv")
        self.assertEqual(list(content.index), [i for i in range(4)])

        shutil.rmtree(path)
