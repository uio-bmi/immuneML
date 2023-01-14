import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.reports.encoding_reports.NonMotifSequenceSimilarity import NonMotifSequenceSimilarity
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestNonMotifSequenceSimilarity(TestCase):
    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_encoded_data(self, path):
        dataset = RandomDatasetGenerator.generate_sequence_dataset(10, {10: 1}, {"l1": {"A": 0.5, "B": 0.5}},
                                                                   path / "dataset")

        lc = LabelConfiguration()
        lc.add_label("l1", ["A", "B"], positive_class="A")

        encoder = MotifEncoder.build_object(
            dataset,
            **{
                "max_positions": 1,
                "min_precision": 0.1,
                "min_recall": 0,
                "min_true_positives": 1,
                "generalize_motifs": False,
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
        PathBuilder.build(path)

        encoded_dataset = self._create_dummy_encoded_data(path)

        report = NonMotifSequenceSimilarity.build_object(
            **{"dataset": encoded_dataset, "result_path": path}
        )

        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        self.assertTrue(os.path.isfile(path / "sequence_hamming_distances.html"))
        self.assertTrue(os.path.isfile(path / "sequence_hamming_distances_percentage.tsv"))
        self.assertTrue(os.path.isfile(path / "sequence_hamming_distances_raw.tsv"))

        shutil.rmtree(path)
